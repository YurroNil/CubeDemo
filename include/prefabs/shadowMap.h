// include/prefabs/shadowMap.h
#pragma once
#include <glad/glad.h>
#include "prefabs/light.h"
#include "graphics/shader.h"

namespace CubeDemo::Prefabs {

class ShadowMap {
public:
    ShadowMap(int width = 1024, int height = 1024);

    void BindForWriting();
    void BindForReading(GLenum textureUnit);
    mat4 GetLightSpaceMat(DL* sun) const;
    void RenderShadow(Camera* camera, const Light& light);
    void CreateShader();

private:
    GLuint m_FBO, m_ShadowMap;
    int m_Width, m_Height;
    Shader* m_ShadowShader = nullptr;
};
}