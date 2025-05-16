// include/graphics/shadowMap.h
#pragma once
#include <glad/glad.h>
#include "graphics/light.h"

namespace CubeDemo::Graphics {

class ShadowMap {
public:
    ShadowMap(int width = 1024, int height = 1024);

    void BindForWriting();
    void BindForReading(GLenum textureUnit);
    mat4 GetLightSpaceMatrix(Graphics::DirLight* sun) const;

private:
    GLuint m_FBO;
    GLuint m_ShadowMap;
    int m_Width, m_Height;
};

}