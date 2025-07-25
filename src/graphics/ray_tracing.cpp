// src/graphics/ray_tracing.cpp
#include "pch.h"
#include "graphics/ray_tracing.h"
#include "resources/model.h"
#include "resources/material.h"
#include "managers/model.h"

namespace CubeDemo {

extern Managers::ModelMng* MODEL_MNG; 
extern std::vector<Model*> MODEL_POINTERS; 
extern bool RT_DEBUG;

RayTracing::RayTracing() 
    : m_OptixBackend(OptixBackend::GetInstance()) {}

RayTracing::~RayTracing() {
    Cleanup();
}

void RayTracing::Cleanup() {
    m_OptixBackend.Shutdown();
}

void RayTracing::Init() {
    int width = WINDOW::GetWidth();
    int height = WINDOW::GetHeight();
    
    // 初始化OptiX
    m_OptixBackend.Init(width, height);
    
    // 创建并上传场景
    CreateSceneBuffers();
}

void RayTracing::CreateSceneBuffers() {
    std::vector<OptixTriangle> triangles;
    std::vector<OptixMaterial> materials;
    
    // 材质索引映射
    std::unordered_map<MaterialPtr, int> materialIndexMap;
    
    for (const auto& model : MODEL_POINTERS) {
        for (const auto& mesh : model->GetMeshes()) {
            const auto& vertices = mesh.Vertices;
            const auto& indices = mesh.GetIndices();
            
            // 处理材质
            MaterialPtr material = mesh.GetMaterial();
            if (materialIndexMap.find(material) == materialIndexMap.end()) {
                materialIndexMap[material] = materials.size();
                
                OptixMaterial mat;
                mat.diffuse = make_float3(material->diffuse.x, material->diffuse.y, material->diffuse.z);
                mat.specular = make_float3(material->specular.x, material->specular.y, material->specular.z);
                mat.emission = make_float3(material->emission.x, material->emission.y, material->emission.z);
                mat.shininess = material->shininess;
                mat.opacity = material->opacity;
                
                materials.push_back(mat);
            }
            int matIndex = materialIndexMap[material];
            
            // 提取三角形
            for (size_t i = 0; i < indices.size(); i += 3) {
                OptixTriangle tri;
                const auto& v0 = vertices[indices[i]];
                const auto& v1 = vertices[indices[i+1]];
                const auto& v2 = vertices[indices[i+2]];
                
                tri.v0 = make_float3(v0.Position.x, v0.Position.y, v0.Position.z);
                tri.v1 = make_float3(v1.Position.x, v1.Position.y, v1.Position.z);
                tri.v2 = make_float3(v2.Position.x, v2.Position.y, v2.Position.z);
                
                tri.n0 = make_float3(v0.Normal.x, v0.Normal.y, v0.Normal.z);
                tri.n1 = make_float3(v1.Normal.x, v1.Normal.y, v1.Normal.z);
                tri.n2 = make_float3(v2.Normal.x, v2.Normal.y, v2.Normal.z);
                
                tri.uv0 = make_float2(v0.TexCoords.x, v0.TexCoords.y);
                tri.uv1 = make_float2(v1.TexCoords.x, v1.TexCoords.y);
                tri.uv2 = make_float2(v2.TexCoords.x, v2.TexCoords.y);
                
                tri.materialIndex = matIndex;
                tri.emission = make_float3(material->emission.x, material->emission.y, material->emission.z);
                
                triangles.push_back(tri);
            }
        }
    }
    
    // 上传场景到OptiX
    m_OptixBackend.UploadScene(triangles, materials);
}

void RayTracing::Render(Camera* camera) {
    // 更新相机
    m_OptixBackend.UpdateCamera(
        camera->Position,
        camera->Position + camera->direction.front,
        camera->direction.up,
        camera->attribute.zoom,
        static_cast<float>(WINDOW::GetWidth())/WINDOW::GetHeight()
    );
    
    // 执行渲染
    m_OptixBackend.Render();
    
    // 使用OptiX的输出纹理
    m_OutputTexture = m_OptixBackend.GetOutputTexture();
}

void RayTracing::RenderDebug(Camera* camera) {
    // 调试模式下使用简单渲染
    Render(camera);
}

unsigned int RayTracing::GetOutputTexture() const { 
    return m_OptixBackend.GetOutputTexture();
}
} // namespace CubeDemo
