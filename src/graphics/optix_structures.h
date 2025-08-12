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
