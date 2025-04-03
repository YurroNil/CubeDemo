// src/renderer/text.cpp

#include <ft2build.h>
#include FT_FREETYPE_H
#include "utils/streams.h"
#include "graphics/textRenderer.h"


// 文本渲染器初始化
void TextRenderer::Init() {
    // 初始化着色器程序（顶点/片段着色器）
    s_TextShader = new Shader("../res/shaders/core/text.vsh", "../res/shaders/core/text.fsh");
    
    // 初始化FreeType库
    FT_Library ft;
    if (FT_Init_FreeType(&ft))
        std::cerr << "FreeType初始化失败" << std::endl;
 
    // 创建字体面对象
    FT_Face face;
    if (FT_New_Face(ft, Font_Simhei, 0, &face))
        std::cerr << "字体加载失败" << std::endl;
 
    // 设置字体像素大小（48号字体）
    FT_Set_Pixel_Sizes(face, 0, 48);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // 设置像素存储对齐方式
 
    // 预加载所有Unicode字符纹理（0x0000-0xFFFF）
    for (uint32_t c = 0; c < 0xFFFF; c++) {
        // 加载字符轮廓数据（跳过加载失败的字符）
        if (FT_Load_Char(face, c, FT_LOAD_RENDER) && FT_Get_Char_Index(face, c) == 0) continue;
        
        // 创建纹理对象
        unsigned int texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        // 设置纹理参数
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RED,          // 使用红色通道存储灰度数据
            face->glyph->bitmap.width,         // 纹理宽度
            face->glyph->bitmap.rows,          // 纹理高度
            0, GL_RED, GL_UNSIGNED_BYTE,       // 数据格式
            face->glyph->bitmap.buffer         // 字体位图数据
        );
 
        // 设置纹理采样参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
 
        // 存储字符信息到映射表
        Character character = {
            texture,                            // 纹理ID
            glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),  // 纹理尺寸
            glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),    // 基准位置偏移
            static_cast<unsigned int>(face->glyph->advance.x)                 // 字符水平间距
        };
        s_Characters.insert(std::make_pair(c, character));
    }
 
    // 清理FreeType资源
    FT_Done_Face(face);
    FT_Done_FreeType(ft);
 
    // 配置顶点数组对象(VAO)和顶点缓冲对象(VBO)
    glGenVertexArrays(1, &s_VAO);
    glGenBuffers(1, &s_VBO);
    glBindVertexArray(s_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW); // 预分配空间
    
    // 设置顶点属性指针（位置+纹理坐标）
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
 
    // 启用混合（用于处理字体边缘抗锯齿）
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
 
// 文本渲染函数
void TextRenderer::RenderText(
    const std::wstring& text,    // 文本
    float x, float y,            // 屏幕位置坐标
    float scale,                 // 文本尺寸
    const vec3& color,           // 文本颜色(默认为白色)
    GLFWwindow* window)           // 窗口指针
{
    // 保存当前OpenGL状态
    GLint lastShader, lastVAO, lastVBO;
    glGetIntegerv(GL_CURRENT_PROGRAM, &lastShader);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &lastVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &lastVBO);
 
    // 设置渲染状态
    glDisable(GL_DEPTH_TEST);  // 禁用深度测试防止文字被遮挡
    glEnable(GL_BLEND);        // 启用混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
 
    // 使用文本着色器程序
    s_TextShader->Use();
    
    // 获取窗口尺寸并设置正交投影矩阵
    WindowManager::UpdateWindowSize(window);
    s_TextShader->SetMat4("projection", glm::ortho(
        0.0f,
        (float)WindowManager::s_WindowWidth,
        0.0f,
        (float)WindowManager::s_WindowHeight)
    );

    s_TextShader->SetVec3("textColor", color); // 设置文字颜色
 
    // 绑定纹理单元和VAO
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(s_VAO);
 
    // 遍历每个字符进行渲染
    for (auto c = text.begin(); c != text.end(); c++) {
        Character ch = s_Characters[*c]; // 获取字符信息
 
        // 计算字符位置（考虑基线偏移）
        float xpos = x + ch.Bearing.x * scale;
        float ypos = y - (ch.Size.y - ch.Bearing.y) * scale;
 
        // 计算字符缩放后的尺寸
        float w = ch.Size.x * scale;
        float h = ch.Size.y * scale;
 
        // 更新VBO数据（6个顶点，每个顶点包含位置和纹理坐标）
        float vertices[6][4] = {
            { xpos,     ypos + h, 0.0f, 0.0f },  // 左上
            { xpos,     ypos,     0.0f, 1.0f },  // 左下
            { xpos + w, ypos,     1.0f, 1.0f },  // 右下
            { xpos,     ypos + h, 0.0f, 0.0f },  // 左上
            { xpos + w, ypos,     1.0f, 1.0f },  // 右下
            { xpos + w, ypos + h, 1.0f, 0.0f }   // 右上
        };
 
        // 绑定纹理并更新VBO数据
        glBindTexture(GL_TEXTURE_2D, ch.TextureID);
        glBindBuffer(GL_ARRAY_BUFFER, s_VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6); // 绘制两个三角形组成四边形
 
        // 更新光标位置（位偏移转换为像素值：>>6 等价于 /64）
        x += (ch.Advance >> 6) * scale;
    }
 
    // 恢复OpenGL状态
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(lastShader);
    glBindVertexArray(lastVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lastVBO);
}