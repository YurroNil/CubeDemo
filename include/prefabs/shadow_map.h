// include/prefabs/shadow_map.h
#pragma once
#include "graphics/shader.h"

namespace CubeDemo::Prefabs {
class ShadowMap {
public:
    ShadowMap(int width = 1024, int height = 1024);

    void BindForWriting();
    void BindForReading(GLenum textureUnit);
    mat4 GetLightSpaceMat(DL* sun) const;
    void RenderShadow(Camera* camera, const Light& light);
    
    // 创建/删除 阴影
    static ShadowMap* CreateShadow(); static void DeleteShadow(ShadowMap* &ptr);
    // 创建/删除 着色器
    void CreateShader(); void DeleteShader();

private:
    GLuint m_FBO, m_ShadowMap;
    int m_Width, m_Height;
    Shader* m_ShadowShader = nullptr;
};
}
