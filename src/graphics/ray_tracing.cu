// src/graphics/ray_tracing.cu
#include <optix_device.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <cstdio>
#include <cfloat>
#include "utils/math_tools.cuh"
#include "graphics/optix_structures.h"

static __constant__ Params params;

struct RayPayload {
    unsigned int pixel_index;    // 像素索引
    unsigned int depth;          // 光线深度
    unsigned int flags;          // 标志位（保留）
    unsigned int done;           // 是否完成 (0/1)
};

static __forceinline__ __device__ float3 randomCosineDirection(curandState* state) {
    float r1 = curand_uniform(state);
    float r2 = curand_uniform(state);
    float z = sqrtf(1.0f - r2);
    float phi = 2.0f * M_PI * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    return make_float3(x, y, z);
}

static __forceinline__ __device__ void coordinateSystem(const float3& n, float3& u, float3& v) {
    if (fabs(n.x) > fabs(n.y)) {
        float invLen = 1.0f / sqrtf(n.x*n.x + n.z*n.z);
        u = make_float3(-n.z * invLen, 0.0f, n.x * invLen);
    } else {
        float invLen = 1.0f / sqrtf(n.y*n.y + n.z*n.z);
        u = make_float3(0.0f, n.z * invLen, -n.y * invLen);
    }
    v = cross(n, u);
}

static __forceinline__ __device__ RayState getRayState(unsigned int pixel_index);
static __forceinline__ __device__ void saveRayState(unsigned int pixel_index, const RayState& state) {
    params.ray_states[pixel_index] = state;
}

// 光线生成程序
extern "C" __global__ void __raygen__main() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    unsigned int pixel_index = idx.y * dim.x + idx.x;
    
    // 调试：打印设备参数指针
    if (pixel_index == 0) {
        printf("[RayGen] Device params pointer: %p\n", &params);
        printf("[RayGen] Params.handle = %p\n", params.handle);
        printf("[RayGen] Params.eye = (%.2f, %.2f, %.2f)\n", 
               params.eye.x, params.eye.y, params.eye.z);
        printf("[RayGen] Params.lookat = (%.2f, %.2f, %.2f)\n", 
               params.lookat.x, params.lookat.y, params.lookat.z);
        printf("[RayGen] Params.triangleCount = %d\n", params.triangleCount);
        printf("[RayGen] Params.ray_states = %p\n", params.ray_states);
    }
    
    // 初始化随机状态
    curand_init(pixel_index, 0, 0, &params.randState[pixel_index]);
    
    // 获取光线状态 - 添加详细调试
    RayState state = getRayState(pixel_index);

    // 设置payload
    unsigned int payload[4] = {
        pixel_index,    // [0] 像素索引
        0,              // [1] 深度 (初始0)
        0,              // [2] 保留
        0               // [3] 完成标志
    };
    
    // 路径追踪循环
    for (int bounce = 0; bounce < 8; bounce++) {
        if (pixel_index == 0) {
            printf("[RayGen] Launching trace for pixel 0, bounce %d\n", bounce);
            printf("[RayGen] Origin: (%.6f, %.6f, %.6f)\n", 
                   state.origin.x, state.origin.y, state.origin.z);
            printf("[RayGen] Direction: (%.6f, %.6f, %.6f)\n", 
                   state.direction.x, state.direction.y, state.direction.z);
            printf("[RayGen] Throughput: (%.6f, %.6f, %.6f)\n",
                   state.throughput.x, state.throughput.y, state.throughput.z);
        }
        
        optixTrace(
            params.handle,
            state.origin,
            state.direction,
            0.001f,    // tmin
            1e16f,     // tmax
            0.0f,      // ray time
            OptixVisibilityMask(0xFF),
            OPTIX_RAY_FLAG_NONE,
            0,         // SBT offset
            0,         // SBT stride
            0,         // miss index
            payload[0], payload[1], payload[2], payload[3]
        );
        
        // 如果光线完成，提前终止
        if (payload[3]) {
            if (pixel_index == 0) printf("[RayGen] Ray marked done at bounce %d\n", bounce);
            break;
        }
        
        // 更新状态
        state = getRayState(pixel_index);
    }
    
    // 应用色调映射
    float3 hdr = state.color;
    float3 ldr = make_float3(
        1.0f - expf(-hdr.x * 1.0f),
        1.0f - expf(-hdr.y * 1.0f),
        1.0f - expf(-hdr.z * 1.0f)
    );
    
    // 写入输出
    params.outputBuffer[pixel_index] = make_float4(ldr.x, ldr.y, ldr.z, 1.0f);
}

// 获取光线状态 - 添加详细调试信息
static __forceinline__ __device__ RayState getRayState(unsigned int pixel_index) {
    RayState state = params.ray_states[pixel_index];
    
    // 调试：打印当前状态值
    if (pixel_index == 0) {
        printf("[RayState] Checking state for pixel 0\n");
        printf("[RayState] Current origin: (%.6f, %.6f, %.6f)\n", 
               state.origin.x, state.origin.y, state.origin.z);

        const float max_float = FLT_MAX;
        printf("[RayState] FLT_MAX = %f (0x%08X)\n", FLT_MAX, *reinterpret_cast<const unsigned int*>(&max_float));
    }
    
    // 检查是否需要初始化
    const unsigned int flt_max_int = 0x7F800000; // FLT_MAX 的 IEEE 754 表示
    unsigned int origin_x_int = *reinterpret_cast<unsigned int*>(&state.origin.x);
    
    if (origin_x_int == flt_max_int) {
        const uint3 dim = optixGetLaunchDimensions();
        const uint3 idx = optixGetLaunchIndex();
        
        // 计算相机光线
        float2 uv = make_float2(
            (static_cast<float>(idx.x) + 0.5f) / dim.x,
            (static_cast<float>(idx.y) + 0.5f) / dim.y
        );
        
        float3 w = normalize(params.lookat - params.eye);
        float3 u = normalize(cross(params.up, w));
        float3 v = normalize(cross(w, u));
        
        float tan_fov = tanf(params.fov * 0.5f * M_PI / 180.0f);
        
        state.origin = params.eye;
        state.direction = normalize(
            u * ((2 * uv.x - 1) * params.aspect * tan_fov) +
            v * ((1 - 2 * uv.y) * tan_fov) +
            w
        );
        state.color = make_float3(0.0f, 0.0f, 0.0f);
        state.throughput = make_float3(1.0f, 1.0f, 1.0f);
        
        // 调试：打印初始光线
        if (pixel_index == 0) {
            printf("[RayState] INITIALIZING pixel 0\n");
            printf("[RayState] Camera eye: (%.6f, %.6f, %.6f)\n", 
                   params.eye.x, params.eye.y, params.eye.z);
            printf("[RayState] UV: (%.6f, %.6f)\n", uv.x, uv.y);
            printf("[RayState] Calculated direction: (%.6f, %.6f, %.6f)\n", 
                   state.direction.x, state.direction.y, state.direction.z);
            printf("[RayState] w vector: (%.6f, %.6f, %.6f)\n", w.x, w.y, w.z);
            printf("[RayState] u vector: (%.6f, %.6f, %.6f)\n", u.x, u.y, u.z);
            printf("[RayState] v vector: (%.6f, %.6f, %.6f)\n", v.x, v.y, v.z);
        }
        
        // 保存初始状态
        saveRayState(pixel_index, state);
    } else if (pixel_index == 0) {
        printf("[RayState] REUSING existing state for pixel 0\n");
    }
    
    return state;
}

// 未命中程序 (天空盒)
extern "C" __global__ void __miss__radiance() {
    // 解析payload
    unsigned int pixel_index = optixGetPayload_0();
    
    // 获取光线状态
    RayState state = getRayState(pixel_index);
    
    // 简单渐变天空
    float t = 0.5f * (state.direction.y + 1.0f);
    float3 skyColor = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + 
                      t * make_float3(0.5f, 0.7f, 1.0f);
    
    state.color += state.throughput * skyColor;
    
    // 调试输出
    if (pixel_index == 0) {
        printf("[Miss] Pixel 0, ray origin: (%.6f, %.6f, %.6f)\n", 
               state.origin.x, state.origin.y, state.origin.z);
        printf("[Miss] Ray direction: (%.6f, %.6f, %.6f)\n", 
               state.direction.x, state.direction.y, state.direction.z);
        printf("[Miss] Sky color: (%.6f, %.6f, %.6f)\n", 
               skyColor.x, skyColor.y, skyColor.z);
    }

    // 标记光线完成
    optixSetPayload_3(1);  // done = true
    
    // 保存状态
    saveRayState(pixel_index, state);
}

// 最近命中程序
extern "C" __global__ void __closesthit__radiance() {
    // 从Payload获取数据
    unsigned int pixel_index = optixGetPayload_0();
    unsigned int depth = optixGetPayload_1();
    unsigned int done = optixGetPayload_3();
    
    // 获取光线状态
    RayState state = getRayState(pixel_index);
    
    // 获取几何数据
    const int primIndex = optixGetPrimitiveIndex();
    
    // 调试输出
    if (pixel_index == 0) {
        printf("[Hit] Pixel 0 hit triangle %d\n", primIndex);
    }

    // 添加边界检查
    if (primIndex < 0 || primIndex >= params.triangleCount) {
        printf("[Hit] 无效的三角形索引: %d (max: %d)\n", primIndex, params.triangleCount - 1);
        done = 1;
        saveRayState(pixel_index, state);
        optixSetPayload_3(done);
        return;
    }

    const OptixTriangle& tri = params.triangles[primIndex];
    const OptixMaterial& mat = params.materials[tri.materialIndex];
    
    // 计算交点位置
    float t = optixGetRayTmax();

    // 检查 t 值有效性
    if (!isfinite(t) || t <= 0.0f || t > 1e10f) {
        printf("[Hit] 无效的t值: %f\n", t);
        done = 1;
        saveRayState(pixel_index, state);
        optixSetPayload_3(done);
        return;
    }

    float3 hitPoint = state.origin + t * state.direction;
    
    // 计算法线 (重心插值)
    float2 bary = optixGetTriangleBarycentrics();
    float3 normal = (1 - bary.x - bary.y) * tri.n0 +
                    bary.x * tri.n1 +
                    bary.y * tri.n2;

    // 确保法线不为零
    float length_sq = dot(normal, normal);
    if (length_sq < 1e-8f) {
        normal = make_float3(0.0f, 1.0f, 0.0f);
    } else {
        normal = normal / sqrtf(length_sq);
    }
    
    // 确保法线朝向光线方向
    if (dot(normal, state.direction) > 0) {
        normal = -normal;
    }
    
    // 自发光贡献
    state.color += state.throughput * mat.emission;
    
    // 获取当前像素的随机状态
    curandState* rand_state = &params.randState[pixel_index];
    
    // 俄罗斯轮盘赌终止
    float rrProb = 0.8f;
    if (depth >= 3) {
        if (curand_uniform(rand_state) > rrProb) {
            done = 1;
            saveRayState(pixel_index, state);
            optixSetPayload_3(done);
            return;
        }
        state.throughput = state.throughput / rrProb;
    }
    
    // 基于材质属性的散射
    float3 newDir;
    if (curand_uniform(rand_state) < 0.8f) { // 漫反射
        float3 u, v;
        coordinateSystem(normal, u, v);
        float3 localDir = randomCosineDirection(rand_state);
        newDir = localDir.x * u + localDir.y * v + localDir.z * normal;
        state.throughput = state.throughput * mat.diffuse * dot(newDir, normal) * (1.0f / 0.8f);
    } else { // 镜面反射
        newDir = state.direction - 2.0f * dot(state.direction, normal) * normal;
        state.throughput = state.throughput * mat.specular * (1.0f / 0.2f);
    }
    
    // 准备下一段光线
    state.origin = hitPoint + normal * 0.001f;
    state.direction = normalize(newDir);
    
    // 更新状态
    saveRayState(pixel_index, state);
    
    // 更新Payload
    depth++;
    optixSetPayload_1(depth);
}