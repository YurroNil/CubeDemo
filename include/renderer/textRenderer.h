// include/renderer/text.h

#pragma once
#include <map>
#include <glm/glm.hpp>
#include <string>
#include "renderer/shader.h"

struct Character {
    unsigned int TextureID;  // 字形纹理ID
    glm::ivec2   Size;       // 字形尺寸
    glm::ivec2   Bearing;    // 从基线到字形左/上的偏移
    unsigned int Advance;    // 到下一个字形的水平偏移
};

class TextRenderer {
public:
    static void Init();
    static void RenderText(const std::wstring& text, float x, float y, float scale, const glm::vec3& color);
    
private:
    inline static std::map<uint32_t, Character> s_Characters;
    inline static Shader* s_TextShader = nullptr;
    inline static unsigned int s_VAO, s_VBO;
};