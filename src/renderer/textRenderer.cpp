// src/renderer/text.cpp

#include <ft2build.h>
#include FT_FREETYPE_H
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include "renderer/textRenderer.h"


void TextRenderer::Init() {
    // 初始化着色器
    s_TextShader = new Shader("../res/shaders/text.vsh", "../res/shaders/text.fsh");
    
    // 加载字体
    FT_Library ft;
    if (FT_Init_FreeType(&ft))
        std::cerr << "FreeType初始化失败" << std::endl;

    FT_Face face;
    if (FT_New_Face(ft,  "C:/Windows/Fonts/simhei.ttf", 0, &face))
        std::cerr << "字体加载失败" << std::endl;

    FT_Set_Pixel_Sizes(face, 0, 48); // 字号48
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // 加载Unicode
    // 汉字常用范围：uint32_t c = 0x4E00; c <= 0x9FA5

    // 加载所有Unicode字符
    for (uint32_t c = 0; c < 0xFFFF; c++) {
    if (FT_Load_Char(face, c, FT_LOAD_RENDER) && FT_Get_Char_Index(face, c) == 0) continue;
        
        unsigned int texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RED,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0, GL_RED, GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        Character character = {
            texture,
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
            static_cast<unsigned int>(face->glyph->advance.x)
        };
        s_Characters.insert(std::make_pair(c, character));
    }





    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    // 配置VAO/VBO
    glGenVertexArrays(1, &s_VAO);
    glGenBuffers(1, &s_VBO);
    glBindVertexArray(s_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void TextRenderer::RenderText(const std::wstring& text, float x, float y, float scale, const glm::vec3& color) {
     GLint lastShader, lastVAO, lastVBO;
    glGetIntegerv(GL_CURRENT_PROGRAM, &lastShader);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &lastVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &lastVBO);

    // 设置文字渲染所需状态
    glDisable(GL_DEPTH_TEST); // 关闭深度测试避免文字被遮挡
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_TextShader->Use();
    // 根据窗口尺寸调整
    int winWidth, winHeight;
    glfwGetWindowSize(glfwGetCurrentContext(), &winWidth, &winHeight);
    s_TextShader->SetMat4("projection", glm::ortho(0.0f, (float)winWidth, 0.0f, (float)winHeight));
    s_TextShader->SetVec3("textColor", color);

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(s_VAO);

    for (auto c = text.begin(); c != text.end(); c++) {
        Character ch = s_Characters[*c];

        float xpos = x + ch.Bearing.x * scale;
        float ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

        float w = ch.Size.x * scale;
        float h = ch.Size.y * scale;

        // 更新VBO数据
        float vertices[6][4] = {
            { xpos,     ypos + h, 0.0f, 0.0f },
            { xpos,     ypos,     0.0f, 1.0f },
            { xpos + w, ypos,     1.0f, 1.0f },

            { xpos,     ypos + h, 0.0f, 0.0f },
            { xpos + w, ypos,     1.0f, 1.0f },
            { xpos + w, ypos + h, 1.0f, 0.0f }
        };

        glBindTexture(GL_TEXTURE_2D, ch.TextureID);
        glBindBuffer(GL_ARRAY_BUFFER, s_VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // 位偏移转换为像素值
        x += (ch.Advance >> 6) * scale;
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(lastShader);
    glBindVertexArray(lastVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lastVBO);
}