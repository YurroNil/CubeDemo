// src/graphics/ray_tracing.cpp
#include "pch.h"
#include "graphics/ray_tracing.h"
#include "resources/model.h"
#include "resources/material.h"
#include "managers/model/mng.h"

namespace CubeDemo {

extern Managers::ModelMng* MODEL_MNG; extern std::vector<Model*> MODEL_POINTERS; extern bool RT_DEBUG;

RayTracing::RayTracing() {
    // 创建计算着色器
    m_RayTraceShader = new Shader(
        CSH_PATH + string("ray_tracing.glsl"), 
        GL_COMPUTE_SHADER
    );
    
    m_BVHBuilderShader = new Shader(
        CSH_PATH + string("bvh_builder.glsl"),
        GL_COMPUTE_SHADER
    );

    if(!RT_DEBUG) return;
    m_DebugShader = new Shader(
        CSH_PATH + string("rt_debug.glsl"), GL_COMPUTE_SHADER
    );
}

RayTracing::~RayTracing() {
    Cleanup();
}

void RayTracing::Cleanup() {
    // 清理资源
    if (m_OutputTexture) glDeleteTextures(1, &m_OutputTexture);
    if (m_AccumulationTexture) glDeleteTextures(1, &m_AccumulationTexture);
    if (m_BVHBuffer) glDeleteBuffers(1, &m_BVHBuffer);
    if(m_RayTraceShader) {delete m_RayTraceShader; m_RayTraceShader = nullptr;}
    if(m_BVHBuilderShader) {delete m_BVHBuilderShader; m_BVHBuilderShader = nullptr;}
    if(m_DebugShader) {delete m_DebugShader; m_DebugShader = nullptr;}
}

void RayTracing::Init() {
    int width = m_FrameWidth = WINDOW::GetWidth();
    int height = m_FrameHeight = WINDOW::GetHeight();
    
    // 创建输出纹理
    glGenTextures(1, &m_OutputTexture);
    glBindTexture(GL_TEXTURE_2D, m_OutputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // 创建累加纹理
    glGenTextures(1, &m_AccumulationTexture);
    glBindTexture(GL_TEXTURE_2D, m_AccumulationTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    
    // 创建场景数据结构
    CreateSceneBuffers();
    BuildBVH();
}

void RayTracing::CreateSceneBuffers() {
    // 计算总三角形数
    size_t totalTriangles = 0;
    std::vector<TriangleData> triangleData;
    std::vector<MaterialData> materialData;
    
    // 收集所有材质
    std::unordered_map<MaterialPtr, int> materialIndexMap;
    
    for (const auto& model : MODEL_POINTERS) {
        for (const auto& mesh : model->GetMeshes()) {
            const auto& vertices = mesh.Vertices;
            const auto& indices = mesh.GetIndices();
            
            // 处理材质
            MaterialPtr material = mesh.GetMaterial();
            if (materialIndexMap.find(material) == materialIndexMap.end()) {
                materialIndexMap[material] = materialData.size();
                
                MaterialData matData;
                matData.diffuse = material->diffuse;
                matData.specular = material->specular;
                matData.shininess = material->shininess;
                matData.emission = material->emission;
                matData.opacity = material->opacity;
                
                materialData.push_back(matData);
            }
            int matIndex = materialIndexMap[material];
            
            // 提取三角形数据
            for (size_t i = 0; i < indices.size(); i += 3) {
                TriangleData tri;
                tri.v0 = vertices[indices[i]].Position;
                tri.v1 = vertices[indices[i+1]].Position;
                tri.v2 = vertices[indices[i+2]].Position;
                
                tri.n0 = vertices[indices[i]].Normal;
                tri.n1 = vertices[indices[i+1]].Normal;
                tri.n2 = vertices[indices[i+2]].Normal;
                
                tri.uv0 = vertices[indices[i]].TexCoords;
                tri.uv1 = vertices[indices[i+1]].TexCoords;
                tri.uv2 = vertices[indices[i+2]].TexCoords;
                
                tri.materialIndex = matIndex;
                tri.emission = material->emission;
                
                triangleData.push_back(tri);
                totalTriangles++;
            }
        }
    }
    
    // 创建GPU缓冲区
    glGenBuffers(1, &m_BVHBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BVHBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        triangleData.size() * sizeof(TriangleData),
        triangleData.data(),
        GL_STATIC_DRAW
    );
    
    // 创建材质缓冲区
    glGenBuffers(1, &m_MaterialBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_MaterialBuffer);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        materialData.size() * sizeof(MaterialData),
        materialData.data(),
        GL_STATIC_DRAW
    );
}

void RayTracing::BuildBVH() {
    m_BVHBuilderShader->Use();
    
    // 设置SSBO绑定
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_BVHBuffer);
    
    // 分派计算着色器
    GLuint numGroups = (m_FrameWidth * m_FrameHeight + 255) / 256;
    glDispatchCompute(numGroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RayTracing::RenderDebug(Camera* camera) {
    if(!RT_DEBUG) return;

    m_DebugShader->Use();
    m_DebugShader->SetMat4("uView", camera->GetViewMat());
    m_DebugShader->SetMat4("uProj", camera->GetProjectionMat(static_cast<float>(m_FrameWidth)/m_FrameHeight));
    m_DebugShader->SetVec3("uCamera.position", camera->Position);
    m_DebugShader->SetVec3("uCamera.up", camera->direction.up);
    m_DebugShader->SetVec3("uCamera.right", camera->direction.right);
    m_DebugShader->SetVec3("uCamera.front", camera->direction.front);
    m_DebugShader->SetFloat("time", static_cast<float>(glfwGetTime()));
    
    // 绑定输出纹理
    glBindImageTexture(0, m_OutputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    // 绑定场景数据
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_BVHBuffer);
    
    // 分派计算着色器
    GLuint numGroupsX = (m_FrameWidth + 7) / 8;
    GLuint numGroupsY = (m_FrameHeight + 7) / 8;
    glDispatchCompute(numGroupsX, numGroupsY, 1);

    // 绘制全屏四边形
    Renderer::RenderFullscreenQuad();
}

void RayTracing::Render(Camera* camera) {
    if(RT_DEBUG) return;

    m_RayTraceShader->Use();
    // 传递相机参数
    m_RayTraceShader->SetVec3("uCamera.Position", camera->Position);
    m_RayTraceShader->SetMat4("uCamera.View", camera->GetViewMat());
    m_RayTraceShader->SetMat4("uCamera.Proj",  camera->GetProjectionMat(static_cast<float>(m_FrameWidth)/m_FrameHeight));
    
    // 绑定输出纹理
    glBindImageTexture(0, m_OutputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glBindImageTexture(1, m_AccumulationTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    
    // 绑定场景数据
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_BVHBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_MaterialBuffer);

    // 分派计算着色器
    GLuint numGroupsX = (m_FrameWidth + 7) / 8;
    GLuint numGroupsY = (m_FrameHeight + 7) / 8;
    glDispatchCompute(numGroupsX, numGroupsY, 1);
    
    // 确保计算完成
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    m_SampleCount++;

    // 绘制全屏四边形
    Renderer::RenderFullscreenQuad();
}

bool RayTracing::TraceRay(const Ray& ray, HitRecord& rec) const {
    return false;
}
}   // namespace CubeDemo
