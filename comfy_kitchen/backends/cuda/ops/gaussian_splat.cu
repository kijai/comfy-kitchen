/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Gaussian splat kernels: projection, tile binning, SH, compositing, 3DGS SSIM and MCMC position noise.
// SH and compositing adapted from nerfstudio-project/gsplat (Apache-2.0); 8x4 warp subtiles as in Faster-GS.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

constexpr int TILE = 16;
constexpr int BLOCK = TILE * TILE;
constexpr int WARPS = BLOCK / 32;
constexpr int SUBTILE_W = 8;
constexpr int SUBTILE_H = 4;
constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;

struct Subtile {
    int x0, y0;   // origin in pixels
    int i, j;     // this lane's pixel, row major within the subtile
};

__device__ __forceinline__ Subtile subtile_of(int tile, int tiles_x, int warp, int lane)
{
    Subtile s;
    s.x0 = (tile % tiles_x) * TILE + (warp % (TILE / SUBTILE_W)) * SUBTILE_W;
    s.y0 = (tile / tiles_x) * TILE + (warp / (TILE / SUBTILE_W)) * SUBTILE_H;
    s.j = s.x0 + lane % SUBTILE_W;
    s.i = s.y0 + lane / SUBTILE_W;
    return s;
}

// Whether a splat's box [x0, x1) x [y0, y1) reaches the subtile; one that misses fails every alpha test there.
__device__ __forceinline__ bool reaches(const int4 box, const Subtile& s)
{
    return box.x < s.x0 + SUBTILE_W && s.x0 < box.z && box.y < s.y0 + SUBTILE_H && s.y0 < box.w;
}

// A chunk's tile, the last one whose chunks start at or before it, and its entries [start, end).
__device__ __forceinline__ int chunk_span(
    int item, const int32_t* chunk_offsets, const int32_t* tile_offsets, int n_tiles, int chunk, int& start, int& end)
{
    int tile = 0;
    int hi = n_tiles;
    while (hi - tile > 1) {
        const int mid = (tile + hi) / 2;
        if (chunk_offsets[mid] <= item) {
            tile = mid;
        } else {
            hi = mid;
        }
    }
    start = tile_offsets[tile] + (item - chunk_offsets[tile]) * chunk;
    end = min(start + chunk, tile_offsets[tile + 1]);
    return tile;
}

template <int C>
__device__ __forceinline__ void write_pixel(
    int pix, float T, const float (&acc)[C], int32_t last, const float* background,
    float* out, float* out_t, int32_t* last_ids)
{
    out_t[pix] = T;
#pragma unroll
    for (int q = 0; q < C; ++q) {
        out[C * pix + q] = acc[q] + T * background[q];
    }
    last_ids[pix] = last;
}

// Entries [start, end) of a tile's list into this lane's pixel from (T, acc), until it closes; warps walk on their own.
template <int C>
__device__ __forceinline__ void composite_run(
    const float2* __restrict__ means2d, const float3* __restrict__ conics, const float* __restrict__ opacities,
    const float* __restrict__ colors, const int4* __restrict__ boxes, const int32_t* __restrict__ flatten_ids,
    const Subtile& s, int lane, int start, int end, float3* xy_opacity_stage, float3* conic_stage,
    float* color_stage, bool& done, float& T, float (&acc)[C], int32_t& last)
{
    const float px = (float)s.j + 0.5f;
    const float py = (float)s.i + 0.5f;
    for (int base = start; base < end && !__all_sync(FULL_MASK, done); base += 32) {
        const int idx = base + lane;
        bool hit = false;
        if (idx < end) {
            const int32_t g = flatten_ids[idx];
            hit = reaches(boxes[g], s);
            if (hit) {
                const float2 xy = means2d[g];
                xy_opacity_stage[lane] = make_float3(xy.x, xy.y, opacities[g]);
                conic_stage[lane] = conics[g];
                // staged with the rest: a global read per blended splat would stall the walk on it
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    color_stage[lane * C + q] = colors[C * g + q];
                }
            }
        }
        unsigned pending = __ballot_sync(FULL_MASK, hit);
        __syncwarp();
        while (pending != 0u) {
            const int k = __ffs(pending) - 1;
            pending &= pending - 1;
            if (done) {
                continue;
            }
            const float3 conic = conic_stage[k];
            const float3 xy_opac = xy_opacity_stage[k];
            const float dx = xy_opac.x - px;
            const float dy = xy_opac.y - py;
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = fminf(0.999f, xy_opac.z * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }
            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                continue;
            }
            const float vis = alpha * T;
#pragma unroll
            for (int q = 0; q < C; ++q) {
                acc[q] += color_stage[k * C + q] * vis;
            }
            last = base + k;
            T = next_T;
        }
        __syncwarp();
    }
}

// Every chunk of every tile composited by its own block, from T = 1. A chunk keeps (T, channels) per pixel.
template <int C>
__global__ void gaussian_rasterize_chunks_kernel(
    const float2* __restrict__ means2d,
    const float3* __restrict__ conics,
    const float* __restrict__ opacities,
    const float* __restrict__ colors,
    const int4* __restrict__ boxes,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const int32_t* __restrict__ chunk_offsets,
    int n_tiles, int width, int height, int tiles_x, int chunk,
    float* __restrict__ chunk_state,
    int32_t* __restrict__ chunk_last,
    uint8_t* __restrict__ chunk_flags)
{
    const int item = blockIdx.x;
    if (item >= chunk_offsets[n_tiles]) {
        return;
    }
    int start, end;
    const int tile = chunk_span(item, chunk_offsets, tile_offsets, n_tiles, chunk, start, end);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const Subtile s = subtile_of(tile, tiles_x, warp, lane);
    const bool inside = s.i < height && s.j < width;

    __shared__ float3 xy_opacity_stage[WARPS][32];
    __shared__ float3 conic_stage[WARPS][32];
    __shared__ float color_stage[WARPS][32 * C];

    bool done = !inside;
    float T = 1.0f;
    float acc[C] = {};
    int32_t last = -1;
    composite_run<C>(means2d, conics, opacities, colors, boxes, flatten_ids, s, lane, start, end,
                     xy_opacity_stage[warp], conic_stage[warp], color_stage[warp], done, T, acc, last);
    const size_t at = (size_t)item * BLOCK + threadIdx.x;
    float* st = chunk_state + at * (C + 1);
    st[0] = T;
#pragma unroll
    for (int q = 0; q < C; ++q) {
        st[1 + q] = acc[q];
    }
    chunk_last[at] = last;
    chunk_flags[at] = inside && done ? 1 : 0;   // closed inside the chunk
}

// Folds each tile's chunks per pixel, flagging the chunk a pixel closes in for a replay from its entry state.
template <int C>
__global__ void gaussian_rasterize_scan_kernel(
    const float* __restrict__ background,
    const int32_t* __restrict__ chunk_offsets,
    int width, int height, int tiles_x,
    float* __restrict__ chunk_state,
    int32_t* __restrict__ chunk_last,
    uint8_t* __restrict__ chunk_flags,
    float* __restrict__ out,
    float* __restrict__ out_t,
    int32_t* __restrict__ last_ids)
{
    const int tile = blockIdx.x;
    const Subtile s = subtile_of(tile, tiles_x, threadIdx.x / 32, threadIdx.x % 32);
    float T = 1.0f;
    float acc[C] = {};
    int32_t last = 0;
    for (int item = chunk_offsets[tile]; item < chunk_offsets[tile + 1]; ++item) {
        const size_t at = (size_t)item * BLOCK + threadIdx.x;
        float* st = chunk_state + at * (C + 1);
        const bool closed = chunk_flags[at] != 0;
        const bool whole = T == 1.0f || (!closed && T * st[0] > 1e-4f);
        if (whole) {
#pragma unroll
            for (int q = 0; q < C; ++q) {
                acc[q] += T * st[1 + q];
            }
            T *= st[0];
            if (chunk_last[at] >= 0) {
                last = chunk_last[at];
            }
        }
        st[0] = T;
#pragma unroll
        for (int q = 0; q < C; ++q) {
            st[1 + q] = acc[q];
        }
        if (!whole) {
            chunk_last[at] = last;
            chunk_flags[at] = 2;
            return;
        }
        if (closed) {
            break;
        }
    }
    if (s.i < height && s.j < width) {
        write_pixel<C>(s.i * width + s.j, T, acc, last, background, out, out_t, last_ids);
    }
}

// Re-walks a chunk for the pixels the scan found closing inside it, from the state they enter with.
template <int C>
__global__ void gaussian_rasterize_replay_kernel(
    const float2* __restrict__ means2d,
    const float3* __restrict__ conics,
    const float* __restrict__ opacities,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    const int4* __restrict__ boxes,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const int32_t* __restrict__ chunk_offsets,
    const float* __restrict__ chunk_state,
    const int32_t* __restrict__ chunk_last,
    const uint8_t* __restrict__ chunk_flags,
    int n_tiles, int width, int tiles_x, int chunk,
    float* __restrict__ out,
    float* __restrict__ out_t,
    int32_t* __restrict__ last_ids)
{
    const int item = blockIdx.x;
    if (item >= chunk_offsets[n_tiles]) {
        return;
    }
    const size_t at = (size_t)item * BLOCK + threadIdx.x;
    const bool mine = chunk_flags[at] == 2;
    if (!__syncthreads_or(mine)) {
        return;
    }
    int start, end;
    const int tile = chunk_span(item, chunk_offsets, tile_offsets, n_tiles, chunk, start, end);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const Subtile s = subtile_of(tile, tiles_x, warp, lane);

    __shared__ float3 xy_opacity_stage[WARPS][32];
    __shared__ float3 conic_stage[WARPS][32];
    __shared__ float color_stage[WARPS][32 * C];

    bool done = !mine;
    const float* st = chunk_state + at * (C + 1);
    float T = st[0];
    float acc[C];
#pragma unroll
    for (int q = 0; q < C; ++q) {
        acc[q] = st[1 + q];
    }
    int32_t last = chunk_last[at];
    composite_run<C>(means2d, conics, opacities, colors, boxes, flatten_ids, s, lane, start, end,
                     xy_opacity_stage[warp], conic_stage[warp], color_stage[warp], done, T, acc, last);
    if (mine) {
        write_pixel<C>(s.i * width + s.j, T, acc, last, background, out, out_t, last_ids);
    }
}

__device__ __forceinline__ float warp_sum(const cg::thread_block_tile<32>& warp, float v)
{
    return cg::reduce(warp, v, cg::plus<float>());
}

// Block totals of v, left in thread 0's v; `scratch` holds K floats per warp. Every thread must call it.
template <int K>
__device__ __forceinline__ void block_sum(float (&v)[K], float* scratch)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int k = 0; k < K; ++k) {
        v[k] = warp_sum(warp, v[k]);
    }
    if (warp.thread_rank() == 0) {
        for (int k = 0; k < K; ++k) {
            scratch[warp.meta_group_rank() * K + k] = v[k];
        }
    }
    block.sync();
    if (block.thread_rank() == 0) {
        for (int k = 0; k < K; ++k) {
            v[k] = 0.f;
            for (int w = 0; w < (int)warp.meta_group_size(); ++w) {
                v[k] += scratch[w * K + k];
            }
        }
    }
}

template <int C>
__global__ void gaussian_rasterize_bwd_kernel(
    const float2* __restrict__ means2d,
    const float3* __restrict__ conics,
    const float* __restrict__ opacities,
    const float* __restrict__ colors,
    const float* __restrict__ background,
    const int4* __restrict__ boxes,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const float* __restrict__ out,
    const float* __restrict__ out_t,
    const int32_t* __restrict__ last_ids,
    const int32_t* __restrict__ chunk_offsets,
    const float* __restrict__ chunk_state,
    const float* __restrict__ v_out,
    int n_tiles, int width, int height, int tiles_x, int chunk,
    float* __restrict__ v_means2d,
    float* __restrict__ v_conics,
    float* __restrict__ v_opacities,
    float* __restrict__ v_colors)
{
    const int item = blockIdx.x;
    if (item >= chunk_offsets[n_tiles]) {
        return;
    }
    int chunk_start, chunk_end;
    const int tile = chunk_span(item, chunk_offsets, tile_offsets, n_tiles, chunk, chunk_start, chunk_end);
    const int warp_id = threadIdx.x / 32;
    const Subtile s = subtile_of(tile, tiles_x, warp_id, threadIdx.x % 32);
    const float px = (float)s.j + 0.5f;
    const float py = (float)s.i + 0.5f;
    const bool inside = s.i < height && s.j < width;
    // lanes past the image edge read a valid pixel and contribute nothing
    const int pix = min(s.i * width + s.j, width * height - 1);

    __shared__ int32_t id_stage[WARPS][32];
    __shared__ float3 xy_opacity_stage[WARPS][32];
    __shared__ float3 conic_stage[WARPS][32];
    __shared__ float color_stage[WARPS][32 * C];

    const float T_final = out_t[pix];
    float T = T_final;
    float buffer[C] = {};   // what the splats behind the current one composite
    const int32_t bin_final = inside ? last_ids[pix] : -1;
    if (bin_final >= chunk_end) {
        // the pixel blends past this chunk: start where the forward was at its end
        const float* st = chunk_state + ((size_t)item * BLOCK + threadIdx.x) * (C + 1);
        T = st[0];
#pragma unroll
        for (int q = 0; q < C; ++q) {
            buffer[q] = out[C * pix + q] - T_final * background[q] - st[1 + q];
        }
    }
    float v_c[C];
    float v_background = 0.f;
#pragma unroll
    for (int q = 0; q < C; ++q) {
        v_c[q] = v_out[C * pix + q];
        v_background += background[q] * v_c[q];
    }

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    const int lane = warp.thread_rank();
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    // back to front from the warp's last contributing splat; lane 0 holds the furthest back entry
    for (int top = min(warp_bin_final, chunk_end - 1); top >= chunk_start; top -= 32) {
        const int idx = top - lane;
        bool hit = false;
        if (idx >= chunk_start) {
            const int32_t g = flatten_ids[idx];
            hit = reaches(boxes[g], s);
            if (hit) {
                const float2 xy = means2d[g];
                id_stage[warp_id][lane] = g;
                xy_opacity_stage[warp_id][lane] = make_float3(xy.x, xy.y, opacities[g]);
                conic_stage[warp_id][lane] = conics[g];
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    color_stage[warp_id][lane * C + q] = colors[C * g + q];
                }
            }
        }
        unsigned pending = warp.ballot(hit);
        warp.sync();
        while (pending != 0u) {
            const int k = __ffs(pending) - 1;
            pending &= pending - 1;
            bool valid = inside && top - k <= bin_final;
            float alpha = 0.f, opac = 0.f, vis = 0.f, dx = 0.f, dy = 0.f;
            float3 conic = make_float3(0.f, 0.f, 0.f);
            if (valid) {
                conic = conic_stage[warp_id][k];
                const float3 xy_opac = xy_opacity_stage[warp_id][k];
                opac = xy_opac.z;
                dx = xy_opac.x - px;
                dy = xy_opac.y - py;
                const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
                vis = __expf(-sigma);
                alpha = fminf(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }
            if (!warp.any(valid)) {
                continue;
            }

            float c[C];
#pragma unroll
            for (int q = 0; q < C; ++q) {
                c[q] = color_stage[warp_id][k * C + q];
            }
            float v_color_local[C] = {};
            float v_conic_local[3] = {0.f, 0.f, 0.f};
            float v_xy_local[2] = {0.f, 0.f};
            float v_opacity_local = 0.f;
            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = -T_final * ra * v_background;
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    v_color_local[q] = fac * v_c[q];
                    v_alpha += (c[q] * T - buffer[q] * ra) * v_c[q];
                }
                // the 0.999 cap has no gradient
                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local[0] = 0.5f * v_sigma * dx * dx;
                    v_conic_local[1] = v_sigma * dx * dy;
                    v_conic_local[2] = 0.5f * v_sigma * dy * dy;
                    v_xy_local[0] = v_sigma * (conic.x * dx + conic.y * dy);
                    v_xy_local[1] = v_sigma * (conic.y * dx + conic.z * dy);
                    v_opacity_local = vis * v_alpha;
                }
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    buffer[q] += c[q] * fac;
                }
            }

#pragma unroll
            for (int q = 0; q < C; ++q) {
                v_color_local[q] = warp_sum(warp, v_color_local[q]);
            }
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                v_conic_local[q] = warp_sum(warp, v_conic_local[q]);
            }
            v_xy_local[0] = warp_sum(warp, v_xy_local[0]);
            v_xy_local[1] = warp_sum(warp, v_xy_local[1]);
            v_opacity_local = warp_sum(warp, v_opacity_local);
            if (lane == 0) {
                const int32_t g = id_stage[warp_id][k];
#pragma unroll
                for (int q = 0; q < C; ++q) {
                    atomicAdd(v_colors + C * g + q, v_color_local[q]);
                }
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    atomicAdd(v_conics + 3 * g + q, v_conic_local[q]);
                }
                atomicAdd(v_means2d + 2 * g, v_xy_local[0]);
                atomicAdd(v_means2d + 2 * g + 1, v_xy_local[1]);
                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
        warp.sync();
    }
}

// Rotation matrix of a (w, x, y, z) quaternion, used as given.
__device__ __forceinline__ void quat_to_rotation(const float4 q, float* r)
{
    const float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
    r[0] = 1.f - 2.f * (qy * qy + qz * qz);
    r[1] = 2.f * (qx * qy - qw * qz);
    r[2] = 2.f * (qx * qz + qw * qy);
    r[3] = 2.f * (qx * qy + qw * qz);
    r[4] = 1.f - 2.f * (qx * qx + qz * qz);
    r[5] = 2.f * (qy * qz - qw * qx);
    r[6] = 2.f * (qx * qz - qw * qy);
    r[7] = 2.f * (qy * qz + qw * qx);
    r[8] = 1.f - 2.f * (qx * qx + qy * qy);
}

struct Projected {
    float cam[3], z, invz, u, v, uc, vc, tx, ty;
    float j00, j02, j11, j12;
    float rq[9], w[9], m[9];
    float s00, s01, s02, s11, s12, s22;
    float a, b, c, det;
};

// Everything the projection computes per splat, shared by the forward and backward kernels.
__device__ __forceinline__ void project_one(
    const float3 mean, const float3 scale, const float4 q, const float* R, const float* K,
    int width, int height, float dilation, Projected& p)
{
    for (int r = 0; r < 3; ++r) {
        p.cam[r] = R[4 * r] * mean.x + R[4 * r + 1] * mean.y + R[4 * r + 2] * mean.z + R[4 * r + 3];
    }
    p.z = p.cam[2];
    const float fx = K[0], cx = K[2], fy = K[4], cy = K[5];
    p.invz = 1.0f / fmaxf(p.z, 1e-6f);
    p.u = p.cam[0] * p.invz;
    p.v = p.cam[1] * p.invz;
    p.uc = fminf(fmaxf(p.u, -(cx / fx + 0.15f * width / fx)), (width - cx) / fx + 0.15f * width / fx);
    p.vc = fminf(fmaxf(p.v, -(cy / fy + 0.15f * height / fy)), (height - cy) / fy + 0.15f * height / fy);
    p.tx = p.z * p.uc;
    p.ty = p.z * p.vc;
    p.j00 = fx * p.invz;
    p.j02 = -fx * p.tx * p.invz * p.invz;
    p.j11 = fy * p.invz;
    p.j12 = -fy * p.ty * p.invz * p.invz;

    quat_to_rotation(q, p.rq);
    const float s[3] = {scale.x, scale.y, scale.z};
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            p.w[3 * i + k] = p.rq[3 * i + k] * s[k];
        }
    }
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 3; ++k) {
            p.m[3 * r + k] = R[4 * r] * p.w[k] + R[4 * r + 1] * p.w[3 + k] + R[4 * r + 2] * p.w[6 + k];
        }
    }
    const float* m = p.m;
    p.s00 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
    p.s01 = m[0] * m[3] + m[1] * m[4] + m[2] * m[5];
    p.s02 = m[0] * m[6] + m[1] * m[7] + m[2] * m[8];
    p.s11 = m[3] * m[3] + m[4] * m[4] + m[5] * m[5];
    p.s12 = m[3] * m[6] + m[4] * m[7] + m[5] * m[8];
    p.s22 = m[6] * m[6] + m[7] * m[7] + m[8] * m[8];

    // (J cov) J^T
    const float r0 = p.j00 * p.s00 + p.j02 * p.s02;
    const float r1 = p.j00 * p.s01 + p.j02 * p.s12;
    const float r2 = p.j00 * p.s02 + p.j02 * p.s22;
    const float u1 = p.j11 * p.s11 + p.j12 * p.s12;
    const float u2 = p.j11 * p.s12 + p.j12 * p.s22;
    p.a = r0 * p.j00 + r2 * p.j02 + dilation;
    p.b = r1 * p.j11 + r2 * p.j12;
    p.c = u1 * p.j11 + u2 * p.j12 + dilation;
    p.det = p.a * p.c - p.b * p.b;
}

// Thin splats seen edge-on can round to det <= 0; they are culled before the inverse.
__device__ __forceinline__ bool projected_visible(const Projected& p, float near_plane, float far_plane)
{
    return p.z > near_plane && p.z < far_plane && p.det > 0.f;
}

__global__ void gaussian_project_fwd_kernel(
    int n,
    const float3* __restrict__ means, const float3* __restrict__ scales, const float4* __restrict__ quats,
    const float* __restrict__ opacities, const float* __restrict__ viewmat, const float* __restrict__ K,
    int width, int height, float near_plane, float far_plane, float dilation,
    float* __restrict__ z_out, float2* __restrict__ centre_out, float3* __restrict__ conic_out,
    int4* __restrict__ boxes_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Projected p;
    project_one(means[i], scales[i], quats[i], viewmat, K, width, height, dilation, p);
    z_out[i] = p.z;
    if (!projected_visible(p, near_plane, far_plane)) {
        centre_out[i] = make_float2(0.f, 0.f);
        conic_out[i] = make_float3(0.f, 0.f, 0.f);
        boxes_out[i] = make_int4(0, 0, 0, 0);
        return;
    }
    const float cx = K[0] * p.cam[0] * p.invz + K[2];
    const float cy = K[4] * p.cam[1] * p.invz + K[5];
    centre_out[i] = make_float2(cx, cy);
    conic_out[i] = make_float3(p.c / p.det, -p.b / p.det, p.a / p.det);

    // alpha >= 1/255 inside d^T conic d <= 2 ln(255 o): half-widths sqrt(2 ln(255 o)) times the std
    const float reach = sqrtf(2.f * logf(fmaxf(opacities[i] / ALPHA_THRESHOLD, 1.f)));
    const float rx = reach * sqrtf(p.a);
    const float ry = reach * sqrtf(p.c);
    const int x0 = (int)fminf(fmaxf(ceilf(cx - rx - 0.5f), 0.f), (float)width);
    const int y0 = (int)fminf(fmaxf(ceilf(cy - ry - 0.5f), 0.f), (float)height);
    const int x1 = (int)fminf(fmaxf(floorf(cx + rx - 0.5f), -1.f), (float)(width - 1)) + 1;
    const int y1 = (int)fminf(fmaxf(floorf(cy + ry - 0.5f), -1.f), (float)(height - 1)) + 1;
    boxes_out[i] = x1 <= x0 || y1 <= y0 ? make_int4(0, 0, 0, 0) : make_int4(x0, y0, x1, y1);
}

__global__ void gaussian_project_bwd_kernel(
    int n,
    const float3* __restrict__ means, const float3* __restrict__ scales, const float4* __restrict__ quats,
    const float* __restrict__ viewmat, const float* __restrict__ K,
    const float2* __restrict__ v_centre, const float3* __restrict__ v_conic,
    int width, int height, float near_plane, float far_plane, float dilation,
    float3* __restrict__ v_means, float3* __restrict__ v_scales, float4* __restrict__ v_quats,
    float* __restrict__ v_viewmat)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float vR[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float vt[3] = {0.f, 0.f, 0.f};
    if (i < n) {
        Projected p;
        const float3 scale = scales[i];
        project_one(means[i], scale, quats[i], viewmat, K, width, height, dilation, p);
        float3 vm = make_float3(0.f, 0.f, 0.f);
        float3 vs = make_float3(0.f, 0.f, 0.f);
        float4 vq = make_float4(0.f, 0.f, 0.f, 0.f);
        if (projected_visible(p, near_plane, far_plane)) {
            const float fx = K[0], fy = K[4];
            const float2 vc2 = v_centre[i];

            // conic = cov^-1, so dL/dcov = -conic dL/dconic conic, the off-diagonal stored once
            const float ixx = p.c / p.det, ixy = -p.b / p.det, iyy = p.a / p.det;
            const float3 vk = v_conic[i];
            const float va = -(vk.x * ixx * ixx + vk.y * ixx * ixy + vk.z * ixy * ixy);
            const float vb = -(2.f * vk.x * ixx * ixy + vk.y * (ixx * iyy + ixy * ixy) + 2.f * vk.z * ixy * iyy);
            const float vcc = -(vk.x * ixy * ixy + vk.y * ixy * iyy + vk.z * iyy * iyy);

            // 2D covariance -> 3D camera covariance entries -> M
            const float vS00 = va * p.j00 * p.j00;
            const float vS02 = va * 2.f * p.j00 * p.j02 + vb * p.j00 * p.j12;
            const float vS22 = va * p.j02 * p.j02 + vb * p.j02 * p.j12 + vcc * p.j12 * p.j12;
            const float vS01 = vb * p.j00 * p.j11;
            const float vS12 = vb * p.j02 * p.j11 + vcc * 2.f * p.j11 * p.j12;
            const float vS11 = vcc * p.j11 * p.j11;
            const float G[9] = {2.f * vS00, vS01, vS02, vS01, 2.f * vS11, vS12, vS02, vS12, 2.f * vS22};
            float vM[9];
            for (int r = 0; r < 3; ++r) {
                for (int k = 0; k < 3; ++k) {
                    vM[3 * r + k] = G[3 * r] * p.m[k] + G[3 * r + 1] * p.m[3 + k] + G[3 * r + 2] * p.m[6 + k];
                }
            }

            // 2D covariance -> Jacobian -> depth and the clamped view ray
            const float vj00 = va * (2.f * p.j00 * p.s00 + 2.f * p.j02 * p.s02) + vb * (p.j11 * p.s01 + p.j12 * p.s02);
            const float vj02 = va * (2.f * p.j00 * p.s02 + 2.f * p.j02 * p.s22) + vb * (p.j11 * p.s12 + p.j12 * p.s22);
            const float vj11 = vb * (p.j00 * p.s01 + p.j02 * p.s12) + vcc * (2.f * p.j11 * p.s11 + 2.f * p.j12 * p.s12);
            const float vj12 = vb * (p.j00 * p.s02 + p.j02 * p.s22) + vcc * (2.f * p.j11 * p.s12 + 2.f * p.j12 * p.s22);
            float vinvz = fx * vj00 + fy * vj11 - 2.f * fx * p.tx * p.invz * vj02 - 2.f * fy * p.ty * p.invz * vj12
                          + fx * p.cam[0] * vc2.x + fy * p.cam[1] * vc2.y;
            const float vtx = -fx * p.invz * p.invz * vj02;
            const float vty = -fy * p.invz * p.invz * vj12;
            float vcam[3] = {fx * p.invz * vc2.x, fy * p.invz * vc2.y, 0.f};
            float vz = p.uc * vtx + p.vc * vty;
            if (p.u == p.uc) {
                const float vu = p.z * vtx;
                vcam[0] += p.invz * vu;
                vinvz += p.cam[0] * vu;
            }
            if (p.v == p.vc) {
                const float vv = p.z * vty;
                vcam[1] += p.invz * vv;
                vinvz += p.cam[1] * vv;
            }
            if (p.z > 1e-6f) {
                vz -= p.invz * p.invz * vinvz;
            }
            vcam[2] += vz;

            // camera point -> mean; camera point and M = R W -> view matrix
            const float* R = viewmat;
            vm = make_float3(R[0] * vcam[0] + R[4] * vcam[1] + R[8] * vcam[2],
                             R[1] * vcam[0] + R[5] * vcam[1] + R[9] * vcam[2],
                             R[2] * vcam[0] + R[6] * vcam[1] + R[10] * vcam[2]);
            const float3 mean = means[i];
            const float mm[3] = {mean.x, mean.y, mean.z};
            for (int r = 0; r < 3; ++r) {
                vt[r] = vcam[r];
                for (int c = 0; c < 3; ++c) {
                    vR[3 * r + c] = vcam[r] * mm[c] + vM[3 * r] * p.w[3 * c] + vM[3 * r + 1] * p.w[3 * c + 1]
                                    + vM[3 * r + 2] * p.w[3 * c + 2];
                }
            }

            // M -> W = Rq diag(s) -> scales and the quaternion rotation
            const float s[3] = {scale.x, scale.y, scale.z};
            float vRq[9];
            float vsk[3] = {0.f, 0.f, 0.f};
            for (int a = 0; a < 3; ++a) {
                for (int k = 0; k < 3; ++k) {
                    const float vw = R[a] * vM[k] + R[4 + a] * vM[3 + k] + R[8 + a] * vM[6 + k];
                    vRq[3 * a + k] = vw * s[k];
                    vsk[k] += vw * p.rq[3 * a + k];
                }
            }
            vs = make_float3(vsk[0], vsk[1], vsk[2]);
            const float4 q = quats[i];
            const float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
            vq.x = 2.f * (-qz * vRq[1] + qy * vRq[2] + qz * vRq[3] - qx * vRq[5] - qy * vRq[6] + qx * vRq[7]);
            vq.y = 2.f * (qy * vRq[1] + qz * vRq[2] + qy * vRq[3] - 2.f * qx * vRq[4] - qw * vRq[5]
                          + qz * vRq[6] + qw * vRq[7] - 2.f * qx * vRq[8]);
            vq.z = 2.f * (-2.f * qy * vRq[0] + qx * vRq[1] + qw * vRq[2] + qx * vRq[3] + qz * vRq[5]
                          - qw * vRq[6] + qz * vRq[7] - 2.f * qy * vRq[8]);
            vq.w = 2.f * (-2.f * qz * vRq[0] - qw * vRq[1] + qx * vRq[2] + qw * vRq[3] - 2.f * qz * vRq[4]
                          + qy * vRq[5] + qx * vRq[6] + qy * vRq[7]);
        }
        v_means[i] = vm;
        v_scales[i] = vs;
        v_quats[i] = vq;
    }

    // the view matrix gradient sums over every splat: per block first, then one atomic per entry
    __shared__ float scratch[WARPS * 12];
    float v_rows[12];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            v_rows[4 * r + c] = vR[3 * r + c];
        }
        v_rows[4 * r + 3] = vt[r];
    }
    block_sum(v_rows, scratch);
    if (threadIdx.x == 0) {
        for (int k = 0; k < 12; ++k) {
            atomicAdd(v_viewmat + k, v_rows[k]);
        }
    }
}

// The 16x16 tiles a pixel box spans, end exclusive; none for an empty box.
__device__ __forceinline__ int4 box_tiles(const int4 box)
{
    return make_int4(box.x / TILE, box.y / TILE, (box.z + TILE - 1) / TILE, (box.w + TILE - 1) / TILE);
}

__global__ void gaussian_bin_count_kernel(
    int n, const int64_t* __restrict__ order, const int4* __restrict__ boxes, int32_t* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int4 r = box_tiles(boxes[order[i]]);
    counts[i] = (r.z - r.x) * (r.w - r.y);
}

// Splat i of the depth order writes its (tile, id) pairs ending at ends[i]; a stable sort by tile finishes.
template <typename key_t>
__global__ void gaussian_bin_kernel(
    int n, const int64_t* __restrict__ order, const int4* __restrict__ boxes, const int32_t* __restrict__ ends,
    int tiles_x, key_t* __restrict__ keys, int32_t* __restrict__ ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int32_t g = (int32_t)order[i];
    const int4 r = box_tiles(boxes[g]);
    int pos = i == 0 ? 0 : ends[i - 1];
    for (int ty = r.y; ty < r.w; ++ty) {
        for (int tx = r.x; tx < r.z; ++tx) {
            keys[pos] = (key_t)(ty * tiles_x + tx);
            ids[pos] = g;
            ++pos;
        }
    }
}

// Real SH to degree 3 at a unit direction, in the 3DGS band order.
__device__ __forceinline__ void sh_basis(int degree, float x, float y, float z, float* b)
{
    b[0] = 0.2820947917738781f;
    if (degree < 1) {
        return;
    }
    b[1] = -0.48860251190292f * y;
    b[2] = 0.48860251190292f * z;
    b[3] = -0.48860251190292f * x;
    if (degree < 2) {
        return;
    }
    const float z2 = z * z;
    const float fTmp0B = -1.092548430592079f * z;
    const float fC1 = x * x - y * y;
    const float fS1 = 2.f * x * y;
    b[4] = 0.5462742152960395f * fS1;
    b[5] = fTmp0B * y;
    b[6] = 0.9461746957575601f * z2 - 0.3153915652525201f;
    b[7] = fTmp0B * x;
    b[8] = 0.5462742152960395f * fC1;
    if (degree < 3) {
        return;
    }
    const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    const float fTmp1B = 1.445305721320277f * z;
    const float fC2 = x * fC1 - y * fS1;
    const float fS2 = x * fS1 + y * fC1;
    b[9] = -0.5900435899266435f * fS2;
    b[10] = fTmp1B * fS1;
    b[11] = fTmp0C * y;
    b[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    b[13] = fTmp0C * x;
    b[14] = fTmp1B * fC1;
    b[15] = -0.5900435899266435f * fC2;
}

// Gradient of sum_k w[k] * basis_k with respect to the unit direction (x, y, z).
__device__ __forceinline__ float3 sh_basis_grad(int degree, float x, float y, float z, const float* w)
{
    float gx = -0.48860251190292f * w[3];
    float gy = -0.48860251190292f * w[1];
    float gz = 0.48860251190292f * w[2];
    if (degree < 2) {
        return make_float3(gx, gy, gz);
    }
    const float z2 = z * z;
    const float fTmp0B = -1.092548430592079f * z;
    const float fC1 = x * x - y * y;
    const float fS1 = 2.f * x * y;
    gx += 0.5462742152960395f * (2.f * y * w[4] + 2.f * x * w[8]) + fTmp0B * w[7];
    gy += 0.5462742152960395f * (2.f * x * w[4] - 2.f * y * w[8]) + fTmp0B * w[5];
    gz += -1.092548430592079f * (y * w[5] + x * w[7]) + 2.f * 0.9461746957575601f * z * w[6];
    if (degree < 3) {
        return make_float3(gx, gy, gz);
    }
    const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    const float fTmp1B = 1.445305721320277f * z;
    const float fC1_x = 2.f * x, fC1_y = -2.f * y, fS1_x = 2.f * y, fS1_y = 2.f * x;
    const float fC2_x = fC1 + x * fC1_x - y * fS1_x;
    const float fC2_y = x * fC1_y - fS1 - y * fS1_y;
    const float fS2_x = fS1 + x * fS1_x + y * fC1_x;
    const float fS2_y = x * fS1_y + fC1 + y * fC1_y;
    const float fTmp0C_z = -2.285228997322329f * 2.f * z;
    const float fTmp1B_z = 1.445305721320277f;
    gx += -0.5900435899266435f * (fS2_x * w[9] + fC2_x * w[15]) + fTmp1B * (fS1_x * w[10] + fC1_x * w[14])
          + fTmp0C * w[13];
    gy += -0.5900435899266435f * (fS2_y * w[9] + fC2_y * w[15]) + fTmp1B * (fS1_y * w[10] + fC1_y * w[14])
          + fTmp0C * w[11];
    gz += fTmp1B_z * (fS1 * w[10] + fC1 * w[14]) + fTmp0C_z * (y * w[11] + x * w[13])
          + (3.f * 1.865881662950577f * z2 - 1.119528997770346f) * w[12];
    return make_float3(gx, gy, gz);
}

// Coefficients go through shared memory, coalesced, with an odd row stride against bank conflicts.
constexpr int SH_BLOCK = 128;

// `width` floats of each of `count` rows lying `stride` floats apart, into the stage rows at column `col`.
__device__ __forceinline__ void sh_stage_load(const float* src, int count, int width, int stride, int srow, int col,
                                              float* stage)
{
    for (int k = threadIdx.x; k < count * width; k += SH_BLOCK) {
        const int g = k / width;
        const int j = k - g * width;
        stage[g * srow + col + j] = src[(size_t)g * stride + j];
    }
}

__global__ void gaussian_sh_fwd_kernel(
    int n, int rest, int degree,
    const float3* __restrict__ means, const float* __restrict__ campos, const float* __restrict__ sh0,
    const float* __restrict__ sh_rest, float* __restrict__ colors)
{
    __shared__ float stage[SH_BLOCK * 49];
    const int base = blockIdx.x * SH_BLOCK;
    const int count = min(SH_BLOCK, n - base);
    const int bands = (degree + 1) * (degree + 1);
    const int srow = (3 * bands) | 1;
    sh_stage_load(sh0 + (size_t)base * 3, count, 3, 3, srow, 0, stage);
    sh_stage_load(sh_rest + (size_t)base * 3 * rest, count, 3 * (bands - 1), 3 * rest, srow, 3, stage);
    __syncthreads();
    if ((int)threadIdx.x >= count) {
        return;
    }
    const int i = base + threadIdx.x;
    const float3 m = means[i];
    const float3 d = make_float3(m.x - campos[0], m.y - campos[1], m.z - campos[2]);
    const float inorm = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float b[16];
    sh_basis(degree, d.x * inorm, d.y * inorm, d.z * inorm, b);
    const float* c = stage + threadIdx.x * srow;
    float rgb[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < bands; ++k) {
        rgb[0] += b[k] * c[3 * k];
        rgb[1] += b[k] * c[3 * k + 1];
        rgb[2] += b[k] * c[3 * k + 2];
    }
    colors[3 * i] = rgb[0];
    colors[3 * i + 1] = rgb[1];
    colors[3 * i + 2] = rgb[2];
}

__global__ void gaussian_sh_bwd_kernel(
    int n, int rest, int degree,
    const float3* __restrict__ means, const float* __restrict__ campos, const float* __restrict__ sh0,
    const float* __restrict__ sh_rest, const float* __restrict__ v_colors, float* __restrict__ v_sh0,
    float* __restrict__ v_sh_rest, float3* __restrict__ v_means, float* __restrict__ v_campos)
{
    __shared__ float stage[SH_BLOCK * 49];
    const int base = blockIdx.x * SH_BLOCK;
    const int count = min(SH_BLOCK, n - base);
    const int bands = (degree + 1) * (degree + 1);
    const int srow = (3 * bands) | 1;
    sh_stage_load(sh0 + (size_t)base * 3, count, 3, 3, srow, 0, stage);
    sh_stage_load(sh_rest + (size_t)base * 3 * rest, count, 3 * (bands - 1), 3 * rest, srow, 3, stage);
    __syncthreads();
    float v_cam[3] = {0.f, 0.f, 0.f};
    if ((int)threadIdx.x < count) {
        const int i = base + threadIdx.x;
        const float3 m = means[i];
        const float3 d = make_float3(m.x - campos[0], m.y - campos[1], m.z - campos[2]);
        const float inorm = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
        const float x = d.x * inorm, y = d.y * inorm, z = d.z * inorm;
        float b[16];
        sh_basis(degree, x, y, z, b);
        const float vc[3] = {v_colors[3 * i], v_colors[3 * i + 1], v_colors[3 * i + 2]};
        // each thread swaps its staged coefficients for their gradients; w[k] feeds the direction gradient
        float* c = stage + threadIdx.x * srow;
        float w[16];
        for (int k = 0; k < bands; ++k) {
            w[k] = vc[0] * c[3 * k] + vc[1] * c[3 * k + 1] + vc[2] * c[3 * k + 2];
            c[3 * k] = b[k] * vc[0];
            c[3 * k + 1] = b[k] * vc[1];
            c[3 * k + 2] = b[k] * vc[2];
        }
        if (degree < 1) {
            v_means[i] = make_float3(0.f, 0.f, 0.f);
        } else {
            // through the normalisation: only the part of the gradient across the direction survives
            const float3 g = sh_basis_grad(degree, x, y, z, w);
            const float along = g.x * x + g.y * y + g.z * z;
            const float3 vd = make_float3((g.x - along * x) * inorm, (g.y - along * y) * inorm, (g.z - along * z) * inorm);
            v_means[i] = vd;
            v_cam[0] = -vd.x;
            v_cam[1] = -vd.y;
            v_cam[2] = -vd.z;
        }
    }
    __shared__ float scratch[SH_BLOCK / 32 * 3];
    block_sum(v_cam, scratch);
    if (threadIdx.x == 0) {
        atomicAdd(v_campos, v_cam[0]);
        atomicAdd(v_campos + 1, v_cam[1]);
        atomicAdd(v_campos + 2, v_cam[2]);
    }
    float* dst0 = v_sh0 + (size_t)base * 3;
    for (int k = threadIdx.x; k < count * 3; k += SH_BLOCK) {
        dst0[k] = stage[(k / 3) * srow + k % 3];
    }
    // bands above the degree in use get zero gradients
    const int used = 3 * (bands - 1);
    float* dst = v_sh_rest + (size_t)base * 3 * rest;
    for (int k = threadIdx.x; k < count * 3 * rest; k += SH_BLOCK) {
        const int g = k / (3 * rest);
        const int j = k - g * 3 * rest;
        dst[k] = j < used ? stage[g * srow + 3 + j] : 0.f;
    }
}

constexpr int SSIM_RADIUS = 5;
constexpr int SSIM_SPAN = TILE + 2 * SSIM_RADIUS;
constexpr float SSIM_C1 = 1e-4f;   // (0.01 L)^2 and (0.03 L)^2 for a dynamic range L of 1
constexpr float SSIM_C2 = 9e-4f;
__constant__ float SSIM_WINDOW[2 * SSIM_RADIUS + 1] = {
    0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f,
    0.21300554275512695f, 0.26601171493530273f, 0.21300554275512695f, 0.10936068743467331f,
    0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};

// Per pixel SSIM of one channel plane per blockIdx.z, and its partials in img1's mean, E[x^2] and E[xy].
__global__ void ssim_fwd_kernel(
    int height, int width, const float* __restrict__ img1, const float* __restrict__ img2,
    float* __restrict__ ssim_map, float* __restrict__ d_mu, float* __restrict__ d_xx, float* __restrict__ d_xy)
{
    __shared__ float tile1[SSIM_SPAN][SSIM_SPAN];
    __shared__ float tile2[SSIM_SPAN][SSIM_SPAN];
    __shared__ float rows[SSIM_SPAN][TILE][5];
    const size_t plane = (size_t)blockIdx.z * height * width;
    const int ox = blockIdx.x * TILE - SSIM_RADIUS;
    const int oy = blockIdx.y * TILE - SSIM_RADIUS;
    const int tid = threadIdx.y * TILE + threadIdx.x;
    for (int k = tid; k < SSIM_SPAN * SSIM_SPAN; k += BLOCK) {
        const int y = oy + k / SSIM_SPAN, x = ox + k % SSIM_SPAN;
        const bool in = y >= 0 && y < height && x >= 0 && x < width;
        tile1[k / SSIM_SPAN][k % SSIM_SPAN] = in ? img1[plane + (size_t)y * width + x] : 0.f;
        tile2[k / SSIM_SPAN][k % SSIM_SPAN] = in ? img2[plane + (size_t)y * width + x] : 0.f;
    }
    __syncthreads();
    for (int k = tid; k < SSIM_SPAN * TILE; k += BLOCK) {
        const int r = k / TILE, c = k % TILE;
        float s[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
        for (int t = 0; t < 2 * SSIM_RADIUS + 1; ++t) {
            const float w = SSIM_WINDOW[t], a = tile1[r][c + t], b = tile2[r][c + t];
            s[0] += w * a;
            s[1] += w * b;
            s[2] += w * a * a;
            s[3] += w * b * b;
            s[4] += w * a * b;
        }
        for (int q = 0; q < 5; ++q) {
            rows[r][c][q] = s[q];
        }
    }
    __syncthreads();
    const int y = blockIdx.y * TILE + threadIdx.y, x = blockIdx.x * TILE + threadIdx.x;
    if (y >= height || x >= width) {
        return;
    }
    float s[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t < 2 * SSIM_RADIUS + 1; ++t) {
        for (int q = 0; q < 5; ++q) {
            s[q] += SSIM_WINDOW[t] * rows[threadIdx.y + t][threadIdx.x][q];
        }
    }
    const float mu1 = s[0], mu2 = s[1];
    const float sigma1 = s[2] - mu1 * mu1, sigma2 = s[3] - mu2 * mu2, sigma12 = s[4] - mu1 * mu2;
    const float a = 2.f * mu1 * mu2 + SSIM_C1, b = 2.f * sigma12 + SSIM_C2;
    const float c = mu1 * mu1 + mu2 * mu2 + SSIM_C1, d = sigma1 + sigma2 + SSIM_C2;
    const float m = a * b / (c * d);
    const size_t p = plane + (size_t)y * width + x;
    ssim_map[p] = m;
    d_mu[p] = (2.f * mu2 * (b - a) - 2.f * mu1 * m * (d - c)) / (c * d);
    d_xx[p] = -m / d;
    d_xy[p] = 2.f * a / (c * d);
}

// Pulls the per pixel partials back through the window: dL/dx = G * (v d_mu) + 2x G * (v d_xx) + y G * (v d_xy).
__global__ void ssim_bwd_kernel(
    int height, int width, const float* __restrict__ img1, const float* __restrict__ img2,
    const float* __restrict__ v_map, const float* __restrict__ d_mu, const float* __restrict__ d_xx,
    const float* __restrict__ d_xy, float* __restrict__ v_img1)
{
    __shared__ float tile[3][SSIM_SPAN][SSIM_SPAN];
    __shared__ float rows[SSIM_SPAN][TILE][3];
    const size_t plane = (size_t)blockIdx.z * height * width;
    const int ox = blockIdx.x * TILE - SSIM_RADIUS;
    const int oy = blockIdx.y * TILE - SSIM_RADIUS;
    const int tid = threadIdx.y * TILE + threadIdx.x;
    for (int k = tid; k < SSIM_SPAN * SSIM_SPAN; k += BLOCK) {
        const int r = k / SSIM_SPAN, c = k % SSIM_SPAN;
        const int y = oy + r, x = ox + c;
        float g0 = 0.f, g1 = 0.f, g2 = 0.f;
        if (y >= 0 && y < height && x >= 0 && x < width) {
            const size_t p = plane + (size_t)y * width + x;
            const float v = v_map[p];
            g0 = v * d_mu[p];
            g1 = v * d_xx[p];
            g2 = v * d_xy[p];
        }
        tile[0][r][c] = g0;
        tile[1][r][c] = g1;
        tile[2][r][c] = g2;
    }
    __syncthreads();
    for (int k = tid; k < SSIM_SPAN * TILE; k += BLOCK) {
        const int r = k / TILE, c = k % TILE;
        float s[3] = {0.f, 0.f, 0.f};
        for (int t = 0; t < 2 * SSIM_RADIUS + 1; ++t) {
            const float w = SSIM_WINDOW[t];
            s[0] += w * tile[0][r][c + t];
            s[1] += w * tile[1][r][c + t];
            s[2] += w * tile[2][r][c + t];
        }
        for (int q = 0; q < 3; ++q) {
            rows[r][c][q] = s[q];
        }
    }
    __syncthreads();
    const int y = blockIdx.y * TILE + threadIdx.y, x = blockIdx.x * TILE + threadIdx.x;
    if (y >= height || x >= width) {
        return;
    }
    float s[3] = {0.f, 0.f, 0.f};
    for (int t = 0; t < 2 * SSIM_RADIUS + 1; ++t) {
        for (int q = 0; q < 3; ++q) {
            s[q] += SSIM_WINDOW[t] * rows[threadIdx.y + t][threadIdx.x][q];
        }
    }
    const size_t p = plane + (size_t)y * width + x;
    v_img1[p] = s[0] + 2.f * img1[p] * s[1] + img2[p] * s[2];
}

// 3DGS-MCMC exploration: means += R S^2 R^T noise * sigmoid(100 ((1 - o) - 0.995)) * scale.
__global__ void gaussian_position_noise_kernel(
    int n, float3* __restrict__ means, const float3* __restrict__ scales, const float4* __restrict__ quats,
    const float* __restrict__ opacities, const float3* __restrict__ noise, float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float r[9];
    quat_to_rotation(quats[i], r);
    const float3 s = scales[i];
    const float3 e = noise[i];
    const float gate = scale / (1.f + expf(-100.f * ((1.f - opacities[i]) - 0.995f)));
    const float s2[3] = {s.x * s.x * gate, s.y * s.y * gate, s.z * s.z * gate};
    float t[3];
    for (int k = 0; k < 3; ++k) {
        t[k] = (r[k] * e.x + r[3 + k] * e.y + r[6 + k] * e.z) * s2[k];
    }
    float3 m = means[i];
    m.x += r[0] * t[0] + r[1] * t[1] + r[2] * t[2];
    m.y += r[3] * t[0] + r[4] * t[1] + r[5] * t[2];
    m.z += r[6] * t[0] + r[7] * t[1] + r[8] * t[2];
    means[i] = m;
}

// torch.optim.Adam's step on the rows `rows` selects; the other rows keep their parameters and moments.
__global__ void sparse_adam_kernel(
    int n, int row, const bool* __restrict__ rows, float* __restrict__ param, const float* __restrict__ grad,
    float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq,
    float beta1, float beta2, float step_size, float bias_correction2_sqrt, float eps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !rows[i / row]) {
        return;
    }
    const float g = grad[i];
    const float m = beta1 * exp_avg[i] + (1.f - beta1) * g;
    const float v = beta2 * exp_avg_sq[i] + (1.f - beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    param[i] -= step_size * m / (sqrtf(v) / bias_correction2_sqrt + eps);
}

// The forward's state kept for the backward, one float32 buffer: each chunk's per-pixel (T, colour) after its
// entries, then each pixel's final transmittance and last contributor (int32).
template <int C>
struct SavedState {
    float* chunk_state;
    float* out_t;
    int32_t* last_ids;

    SavedState(const void* base, int n_items, int pixels)
        : chunk_state(static_cast<float*>(const_cast<void*>(base))),
          out_t(chunk_state + (size_t)n_items * BLOCK * (C + 1)),
          last_ids(reinterpret_cast<int32_t*>(out_t + pixels)) {}
};

template <int C>
void rasterize_fwd(
    const void* means2d, const void* conics, const void* opacities, const void* colors,
    const void* background, const void* boxes, const void* tile_offsets, const void* flatten_ids,
    const void* chunk_offsets, int n_items, int n_tiles, int width, int height, int tiles_x, int chunk,
    void* out, void* state, void* scratch, cudaStream_t stream)
{
    const SavedState<C> s(state, n_items, width * height);
    // forward-only scratch: each chunk pixel's last contributor, then whether it closed inside the chunk
    int32_t* chunk_last = static_cast<int32_t*>(scratch);
    uint8_t* chunk_flags = reinterpret_cast<uint8_t*>(chunk_last + (size_t)n_items * BLOCK);
    if (n_items > 0) {
        gaussian_rasterize_chunks_kernel<C><<<n_items, BLOCK, 0, stream>>>(
            static_cast<const float2*>(means2d), static_cast<const float3*>(conics),
            static_cast<const float*>(opacities), static_cast<const float*>(colors), static_cast<const int4*>(boxes),
            static_cast<const int32_t*>(tile_offsets), static_cast<const int32_t*>(flatten_ids),
            static_cast<const int32_t*>(chunk_offsets), n_tiles, width, height, tiles_x, chunk,
            s.chunk_state, chunk_last, chunk_flags);
    }
    gaussian_rasterize_scan_kernel<C><<<n_tiles, BLOCK, 0, stream>>>(
        static_cast<const float*>(background), static_cast<const int32_t*>(chunk_offsets), width, height, tiles_x,
        s.chunk_state, chunk_last, chunk_flags, static_cast<float*>(out), s.out_t, s.last_ids);
    if (n_items > 0) {
        gaussian_rasterize_replay_kernel<C><<<n_items, BLOCK, 0, stream>>>(
            static_cast<const float2*>(means2d), static_cast<const float3*>(conics),
            static_cast<const float*>(opacities), static_cast<const float*>(colors),
            static_cast<const float*>(background), static_cast<const int4*>(boxes),
            static_cast<const int32_t*>(tile_offsets), static_cast<const int32_t*>(flatten_ids),
            static_cast<const int32_t*>(chunk_offsets), s.chunk_state, chunk_last, chunk_flags,
            n_tiles, width, tiles_x, chunk, static_cast<float*>(out), s.out_t, s.last_ids);
    }
}

template <int C>
void rasterize_bwd(
    const void* means2d, const void* conics, const void* opacities, const void* colors,
    const void* background, const void* boxes, const void* tile_offsets, const void* flatten_ids,
    const void* out, const void* state, const void* chunk_offsets, const void* v_out,
    int n_items, int n_tiles, int width, int height, int tiles_x, int chunk,
    void* v_means2d, void* v_conics, void* v_opacities, void* v_colors, cudaStream_t stream)
{
    const SavedState<C> s(state, n_items, width * height);
    gaussian_rasterize_bwd_kernel<C><<<n_items, BLOCK, 0, stream>>>(
        static_cast<const float2*>(means2d), static_cast<const float3*>(conics),
        static_cast<const float*>(opacities), static_cast<const float*>(colors),
        static_cast<const float*>(background), static_cast<const int4*>(boxes),
        static_cast<const int32_t*>(tile_offsets), static_cast<const int32_t*>(flatten_ids),
        static_cast<const float*>(out), s.out_t, s.last_ids, static_cast<const int32_t*>(chunk_offsets),
        s.chunk_state, static_cast<const float*>(v_out),
        n_tiles, width, height, tiles_x, chunk,
        static_cast<float*>(v_means2d), static_cast<float*>(v_conics),
        static_cast<float*>(v_opacities), static_cast<float*>(v_colors));
}

} // namespace

// `channels` is 3 or 5; the caller checks.
extern "C" void launch_gaussian_rasterize_fwd(
    const void* means2d, const void* conics, const void* opacities, const void* colors,
    const void* background, const void* boxes, const void* tile_offsets, const void* flatten_ids,
    const void* chunk_offsets, int n_items, int n_tiles, int width, int height, int tiles_x, int chunk, int channels,
    void* out, void* state, void* scratch, cudaStream_t stream)
{
    if (n_tiles == 0) {
        return;
    }
    auto launch = channels == 5 ? rasterize_fwd<5> : rasterize_fwd<3>;
    launch(means2d, conics, opacities, colors, background, boxes, tile_offsets, flatten_ids, chunk_offsets, n_items,
           n_tiles, width, height, tiles_x, chunk, out, state, scratch, stream);
}

extern "C" void launch_gaussian_rasterize_bwd(
    const void* means2d, const void* conics, const void* opacities, const void* colors,
    const void* background, const void* boxes, const void* tile_offsets, const void* flatten_ids,
    const void* out, const void* state, const void* chunk_offsets, const void* v_out,
    int n_items, int n_tiles, int width, int height, int tiles_x, int chunk, int channels,
    void* v_means2d, void* v_conics, void* v_opacities, void* v_colors, cudaStream_t stream)
{
    if (n_items == 0) {
        return;
    }
    auto launch = channels == 5 ? rasterize_bwd<5> : rasterize_bwd<3>;
    launch(means2d, conics, opacities, colors, background, boxes, tile_offsets, flatten_ids, out, state,
           chunk_offsets, v_out, n_items, n_tiles, width, height, tiles_x, chunk,
           v_means2d, v_conics, v_opacities, v_colors, stream);
}

extern "C" void launch_gaussian_project_fwd(
    const void* means, const void* scales, const void* quats, const void* opacities, const void* viewmat,
    const void* K, int n, int width, int height, float near_plane, float far_plane, float dilation,
    void* z, void* centre, void* conic, void* boxes, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_project_fwd_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        n, static_cast<const float3*>(means), static_cast<const float3*>(scales), static_cast<const float4*>(quats),
        static_cast<const float*>(opacities), static_cast<const float*>(viewmat), static_cast<const float*>(K),
        width, height, near_plane, far_plane, dilation, static_cast<float*>(z), static_cast<float2*>(centre),
        static_cast<float3*>(conic), static_cast<int4*>(boxes));
}

extern "C" void launch_gaussian_project_bwd(
    const void* means, const void* scales, const void* quats, const void* viewmat, const void* K,
    const void* v_centre, const void* v_conic,
    int n, int width, int height, float near_plane, float far_plane, float dilation,
    void* v_means, void* v_scales, void* v_quats, void* v_viewmat, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_project_bwd_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        n, static_cast<const float3*>(means), static_cast<const float3*>(scales), static_cast<const float4*>(quats),
        static_cast<const float*>(viewmat), static_cast<const float*>(K),
        static_cast<const float2*>(v_centre), static_cast<const float3*>(v_conic), width, height, near_plane,
        far_plane, dilation, static_cast<float3*>(v_means), static_cast<float3*>(v_scales),
        static_cast<float4*>(v_quats), static_cast<float*>(v_viewmat));
}

extern "C" void launch_gaussian_bin_count(
    const void* order, const void* boxes, int n, void* counts, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_bin_count_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        n, static_cast<const int64_t*>(order), static_cast<const int4*>(boxes), static_cast<int32_t*>(counts));
}

extern "C" void launch_gaussian_bin(
    const void* order, const void* boxes, const void* ends, int n, int tiles_x, int key_bytes,
    void* keys, void* ids, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    if (key_bytes == 2) {
        gaussian_bin_kernel<int16_t><<<(n + 255) / 256, 256, 0, stream>>>(
            n, static_cast<const int64_t*>(order), static_cast<const int4*>(boxes), static_cast<const int32_t*>(ends),
            tiles_x, static_cast<int16_t*>(keys), static_cast<int32_t*>(ids));
    } else {
        gaussian_bin_kernel<int32_t><<<(n + 255) / 256, 256, 0, stream>>>(
            n, static_cast<const int64_t*>(order), static_cast<const int4*>(boxes), static_cast<const int32_t*>(ends),
            tiles_x, static_cast<int32_t*>(keys), static_cast<int32_t*>(ids));
    }
}

extern "C" void launch_gaussian_sh_fwd(
    const void* means, const void* campos, const void* sh0, const void* sh_rest, int n, int rest, int degree,
    void* colors, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_sh_fwd_kernel<<<(n + SH_BLOCK - 1) / SH_BLOCK, SH_BLOCK, 0, stream>>>(
        n, rest, degree, static_cast<const float3*>(means), static_cast<const float*>(campos),
        static_cast<const float*>(sh0), static_cast<const float*>(sh_rest), static_cast<float*>(colors));
}

extern "C" void launch_gaussian_sh_bwd(
    const void* means, const void* campos, const void* sh0, const void* sh_rest, const void* v_colors, int n,
    int rest, int degree, void* v_sh0, void* v_sh_rest, void* v_means, void* v_campos, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_sh_bwd_kernel<<<(n + SH_BLOCK - 1) / SH_BLOCK, SH_BLOCK, 0, stream>>>(
        n, rest, degree, static_cast<const float3*>(means), static_cast<const float*>(campos),
        static_cast<const float*>(sh0), static_cast<const float*>(sh_rest), static_cast<const float*>(v_colors),
        static_cast<float*>(v_sh0), static_cast<float*>(v_sh_rest), static_cast<float3*>(v_means),
        static_cast<float*>(v_campos));
}

extern "C" void launch_ssim_fwd(
    const void* img1, const void* img2, int planes, int height, int width,
    void* ssim_map, void* d_mu, void* d_xx, void* d_xy, cudaStream_t stream)
{
    if (planes == 0 || height == 0 || width == 0) {
        return;
    }
    const dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE, planes);
    ssim_fwd_kernel<<<grid, dim3(TILE, TILE), 0, stream>>>(
        height, width, static_cast<const float*>(img1), static_cast<const float*>(img2),
        static_cast<float*>(ssim_map), static_cast<float*>(d_mu), static_cast<float*>(d_xx),
        static_cast<float*>(d_xy));
}

extern "C" void launch_ssim_bwd(
    const void* img1, const void* img2, const void* v_map, const void* d_mu, const void* d_xx, const void* d_xy,
    int planes, int height, int width, void* v_img1, cudaStream_t stream)
{
    if (planes == 0 || height == 0 || width == 0) {
        return;
    }
    const dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE, planes);
    ssim_bwd_kernel<<<grid, dim3(TILE, TILE), 0, stream>>>(
        height, width, static_cast<const float*>(img1), static_cast<const float*>(img2),
        static_cast<const float*>(v_map), static_cast<const float*>(d_mu), static_cast<const float*>(d_xx),
        static_cast<const float*>(d_xy), static_cast<float*>(v_img1));
}

extern "C" void launch_gaussian_position_noise(
    void* means, const void* scales, const void* quats, const void* opacities, const void* noise, int n,
    float scale, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    gaussian_position_noise_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        n, static_cast<float3*>(means), static_cast<const float3*>(scales), static_cast<const float4*>(quats),
        static_cast<const float*>(opacities), static_cast<const float3*>(noise), scale);
}

extern "C" void launch_sparse_adam(
    void* param, const void* grad, void* exp_avg, void* exp_avg_sq, const void* rows, int n, int row,
    float beta1, float beta2, float step_size, float bias_correction2_sqrt, float eps, cudaStream_t stream)
{
    if (n == 0) {
        return;
    }
    sparse_adam_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        n, row, static_cast<const bool*>(rows), static_cast<float*>(param), static_cast<const float*>(grad),
        static_cast<float*>(exp_avg), static_cast<float*>(exp_avg_sq), beta1, beta2, step_size,
        bias_correction2_sqrt, eps);
}
