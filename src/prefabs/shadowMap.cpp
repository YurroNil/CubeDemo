// src/prefabs/shadowMap.h

#include "prefabs/shadowMap.h"
#include "resources/model.h"
#include "utils/defines.h"

// 外部变量声明
namespace CubeDemo {
    extern Shader* MODEL_SHADER;
    extern std::vector<Model*> MODEL_POINTERS;
}

// 别名
using PSM = CubeDemo::Prefabs::ShadowMap;

namespace CubeDemo::Prefabs {

ShadowMap::ShadowMap(int width, int height) : m_Width(width), m_Height(height) {
    glGenFramebuffers(1, &m_FBO);
    
    // 创建深度纹理
    glGenTextures(1, &m_ShadowMap);
    glBindTexture(GL_TEXTURE_2D, m_ShadowMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_Width, m_Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    // 绑定到FBO
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_ShadowMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ShadowMap::BindForWriting() {
    glViewport(0, 0, m_Width, m_Height);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glClear(GL_DEPTH_BUFFER_BIT); // 深度缓冲清除
}

void ShadowMap::BindForReading(GLenum textureUnit) {
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, m_ShadowMap);
}

mat4 ShadowMap::GetLightSpaceMat(DL* sun) const {

    // 根据场景最大包围球半径动态计算
    float sceneRadius = 15.0f;
    
    // 计算光源方向对齐的AABB
    vec3 center = vec3(0.0f); // 假设场景中心在原点
    float extend = sceneRadius;
    mat4 lightView = glm::lookAt(-sun->direction * extend, center, vec3(0,1,0));
    
    // 动态计算正交投影范围
    mat4 lightProjection = glm::ortho(
        -extend, extend, 
        -extend, extend, 
        0.1f, 
        2.0f * extend
    );
    return lightProjection * lightView;
}

// 渲染深度到阴影贴图
void ShadowMap::RenderShadow(Camera* camera, const Light& light) {
    
    BindForWriting();
    m_ShadowShader->Use();
    
    // 阴影矩阵计算
    const auto lightSpaceMatrix = GetLightSpaceMat(light.Get.DirLight());
    m_ShadowShader->SetMat4("lightSpaceMatrix", lightSpaceMatrix);

    // 简化绘制模型（仅位置属性）
    for (auto* model : MODEL_POINTERS) {
        if (model->IsReady() && camera->isSphereVisible(model->bounds.Center, model->bounds.Rad)) {
            m_ShadowShader->SetMat4("model", model->GetModelMatrix());
            model->DrawSimple();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 传递阴影贴图
    BindForReading(GL_TEXTURE1);
    MODEL_SHADER->SetInt("shadowMap", 1);
    MODEL_SHADER->SetMat4("lightSpaceMatrix", lightSpaceMatrix);
}

// 创建/删除 阴影
ShadowMap* ShadowMap::CreateShadow() {
    return new ShadowMap(2048, 2048);
}
void ShadowMap::DeleteShadow(ShadowMap* &ptr) {
    delete ptr; ptr = nullptr;
}


// 创建/删除 阴影深度着色器
void ShadowMap::CreateShader() {
    Shader* shadow_depth = new Shader(
        VSH_PATH + string("shadow_depth.glsl"),
        FSH_PATH + string("shadow_depth.glsl")
    );
    m_ShadowShader = shadow_depth;
}
void ShadowMap::DeleteShader() {
    delete m_ShadowShader; m_ShadowShader = nullptr;
}
}   // namespace CubeDemo::Graphics
