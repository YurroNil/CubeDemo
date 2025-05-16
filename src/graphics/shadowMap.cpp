// src/graphics/shadowMap.h
#include "graphics/shadowMap.h"

namespace CubeDemo::Graphics {

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
    glClear(GL_DEPTH_BUFFER_BIT);
}

void ShadowMap::BindForReading(GLenum textureUnit) {
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, m_ShadowMap);
}

mat4 ShadowMap::GetLightSpaceMatrix(Graphics::DirLight* sun) const {

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
}   // namespace CubeDemo::Graphics
