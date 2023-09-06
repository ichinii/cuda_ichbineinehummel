#pragma once

__device__ ivec2 clamp_coord(ivec2 coord) {
    return clamp(coord, ivec2(0, 0), WinSize-1);
}

__device__ ivec2 wrap_coord(ivec2 coord) {
    return vec2(coord.x % WinSize.x, coord.y % WinSize.y);
}

__device__ bool is_coord_in_bounds(ivec2 coord) {
    return coord == clamp_coord(coord);
}

__device__ ivec2 index_to_coord(int index) {
    ivec2 coord = ivec2(index % WinSize.x, index / WinSize.x);
    return coord;
}

__device__ int coord_to_index(ivec2 coord) {
    return coord.x + coord.y * WinSize.x;
}

__device__ float sdmin(float a, float b) {
    return abs(a) < abs(b) ? a : b;
}

__device__ float sdf_plane(vec3 p, vec3 n) {
    return dot(p, n);
}

__device__ float sdf_sphere(vec3 p) {
    return length(p);
}

__device__ float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}

// __device__ float erose(float *c, int gid) {
//     vec2 coord = index_to_coord(gid);
//     float r = 0.0f;
//     for (int x = -1; x < 2; ++x) {
//         for (int y = -1; y < 2; ++y) {
//             int i = coord_to_index(coord);
//             r = max(r, c[i]);
//         }
//     }
//     return r;
// }

__device__ int length_squared(ivec2 a) {
    return a.x*a.x + a.y*a.y;
}

__device__ bool binary(float c) {
    return 0.5f < c;
}

__device__ constexpr const int invalid_jfa_id = -1;

__global__ void init_jfa_src(int *jfa_src, float *tex_src) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    jfa_src[gid] = binary(tex_src[gid]) ? gid : invalid_jfa_id;
}

__global__ void jfa(int *jfa_dst, int *jfa_src, int n) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int id_self = gid;
    ivec2 coord_self = index_to_coord(id_self);
    int id_self_deref = jfa_src[id_self];
    int l_self_to_self_deref = length_squared(index_to_coord(id_self_deref) - coord_self);

    for (int y = -1; y < 2; ++y) {
        for (int x = -1; x < 2; ++x) {
            ivec2 coord_it = coord_self + ivec2(x, y) * n;
            coord_it = wrap_coord(coord_it);
            int id_it = coord_to_index(coord_it);
            int id_it_deref = jfa_src[id_it];
            ivec2 coord_it_deref = index_to_coord(id_it_deref);
            int l_self_to_it_deref = length_squared(coord_it_deref - coord_self);

            bool is_id_self_deref_valid = id_self_deref != invalid_jfa_id;
            bool is_id_it_deref_valid = id_it_deref != invalid_jfa_id;
            if (is_id_it_deref_valid && (!is_id_self_deref_valid || l_self_to_it_deref <= l_self_to_self_deref)) {
                l_self_to_self_deref = l_self_to_it_deref;
                id_self_deref = id_it_deref;
            }
        }
    }

    jfa_dst[gid] = id_self_deref;
}

__global__ void jfa_to_sdf(float *jfa_dst, int *jfa_src) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    vec2 coord_self = index_to_coord(gid);
    int id_self_deref = jfa_src[gid];
    bool is_id_self_deref_valid = id_self_deref != invalid_jfa_id;
    vec2 coord_deref = index_to_coord(id_self_deref);
    jfa_dst[gid] = is_id_self_deref_valid ? length(coord_deref - coord_self) : WinDiag;
}

// TODO: 1px diagonal lines should not be crossed
__device__ float march(vec2 ro, vec2 rd, float *sdf) {
    float lo = 0.0f;

    for (int i = 0; i < 100; ++i) {
        int id = coord_to_index(ro);

        float l = sdf[id];

        lo += l;
        ro += l * rd;

        if (l < 0.01f || lo > 1000.0f || !is_coord_in_bounds(ro))
            break;
    }

    return lo;
}

__global__ void dilate(float *dst, float *src) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    bool result = false;
    for (int y = -1; y < 2; ++y) {
        for (int x = -1; x < 2; ++x) {
            ivec2 coord = index_to_coord(gid) + ivec2(x, y);
            coord = clamp_coord(coord);
            int index = coord_to_index(coord);
            result = result || binary(src[index]);
        }
    }

    dst[gid] = (float)result;
}

__global__ void erose(float *dst, float *src) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    bool result = true;
    for (int y = -1; y < 2; ++y) {
        for (int x = -1; x < 2; ++x) {
            ivec2 coord = index_to_coord(gid) + ivec2(x, y);
            coord = clamp_coord(coord);
            int index = coord_to_index(coord);
            result = result && binary(src[index]);
        }
    }

    dst[gid] = (float)result;
}

__global__ void get_image(vec4 *c, float *tex, float *sdf_tex) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    vec3 col = vec3(0);

    col.r = tex[gid];

    float l = sdf_tex[gid];
    col.b = l / WinDiag;

    vec2 ro = index_to_coord(gid);
    vec2 rd = normalize(vec2(WinSize) / 2.0f - ro);
    float lo = march(ro, rd, sdf_tex);
    col.g = lo / (float)WinSize.x;

    c[gid] = vec4(col, 1);
    c[gid] = vec4(pow(vec3(c[gid]), vec3(1.0 / 2.2)), c[gid].a);
}

void draw(vec4 *image, float *tex_in) {
    float *tex_src, *tex_dst;
    cudaMallocManaged(&tex_src, Size * sizeof(float));
    cudaMallocManaged(&tex_dst, Size * sizeof(float));
    cudaMemcpy(tex_src, tex_in, Size * sizeof(float), cudaMemcpyDeviceToDevice);

    float *sdf_src;
    cudaMallocManaged(&sdf_src, Size * sizeof(float));

    {
        int *jfa_src, *jfa_dst;
        cudaMallocManaged(&jfa_src, Size * sizeof(int));
        cudaMallocManaged(&jfa_dst, Size * sizeof(int));

        // cudaMemset(jfa_src, -1, Size * sizeof(int));

        {
            int I = 1;
            for (int i = 0; i < I+1; ++i) {
                dilate<<<B, W>>>(tex_dst, tex_src);
                std::swap(tex_dst, tex_src);
            }
            for (int i = 0; i < I; ++i) {
                erose<<<B, W>>>(tex_dst, tex_src);
                std::swap(tex_dst, tex_src);
            }
        }

        init_jfa_src<<<B, W>>>(jfa_src, tex_src);

        assert(WinSize.x == WinSize.y);
        for (int i = WinSize.x/2; i >= 1; i/=2) {
            jfa<<<B, W>>>(jfa_dst, jfa_src, i);
            std::swap(jfa_dst, jfa_src);
        }
        std::swap(jfa_dst, jfa_src);

        jfa_to_sdf<<<B, W>>>(sdf_src, jfa_dst);

        cudaFree(jfa_src);
        cudaFree(jfa_dst);
    }

    get_image<<<B, W>>>(image, tex_src, sdf_src);

    cudaFree(sdf_src);
    cudaFree(tex_dst);
    cudaFree(tex_src);
}
