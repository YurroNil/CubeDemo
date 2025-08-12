// include/graphics/optix_structures.h
#pragma once

// 定义OptiX使用的数据结构
struct OptixTriangle {
    float3 v0, v1, v2;
    float3 n0, n1, n2;
    float2 uv0, uv1, uv2;
    int materialIndex;
    float3 emission;
};

struct OptixMaterial {
    float3 diffuse;
    float3 specular;
    float3 emission;
    float shininess;
    float opacity;
};

// 光线状态 (存储在全局内存)
struct RayState {
    float3 color;
    float3 throughput;
    float3 origin;
    float3 direction;
};


// 启动参数
struct Params {
    OptixTraversableHandle handle;
    float4* outputBuffer; // 线性设备内存指针
    int width;
    int height;
    float3 eye;
    float3 lookat;
    float3 up;
    float fov;
    float aspect;
    OptixTriangle* triangles;
    OptixMaterial* materials;
    int triangleCount; // 三角形计数
    curandState* randState;
    RayState* ray_states;
};