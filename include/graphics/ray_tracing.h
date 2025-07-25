// include/graphics/ray_tracing.h
#pragma once
#include "resources/fwd.h"
#include "graphics/optix_backend.h"

namespace CubeDemo {

// 定义三角形数据结构
struct TriangleData {
    vec3 v0, v1, v2; // 顶点位置
    vec3 n0, n1, n2; // 顶点法线
    vec2 uv0, uv1, uv2; // 纹理坐标
    int materialIndex; // 材质索引
    vec3 emission; // 自发光颜色
};

struct MaterialData {
    vec3 diffuse, specular, emission;
    float shininess, opacity;
};

struct Ray {
    vec3 origin;
    vec3 direction;
    float tmin = 0.001f;
    float tmax = FLT_MAX;
};

struct HitRecord {
    float t;
    vec3 position;
    vec3 normal;
    MaterialPtr material;
};

class RayTracing {
public:
    RayTracing();
    ~RayTracing();

    void Init();
    void Cleanup();
    void Render(Camera* camera);
    void RenderDebug(Camera* camera);
    
    unsigned int GetOutputTexture() const;

private:
    void CreateSceneBuffers();
    
    OptixBackend& m_OptixBackend;
    unsigned int m_OutputTexture = 0;
};
}   // namespace CubeDemo
