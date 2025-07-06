// src/resources/place_holder.cpp
#include "pch.h"
#include "resources/place_holder.h"

namespace CubeDemo {

namespace {
    // 生成棋盘纹理数据
    std::array<unsigned char, 32 * 32 * 3> GenerateCheckerboardData() {
        std::array<unsigned char, 32 * 32 * 3> data{};
        const unsigned char light[3] = {0xF0, 0xF0, 0xF0};  // 浅色 (浅灰)
        const unsigned char dark[3] = {0x40, 0x40, 0x40};   // 深色 (深灰)
        const int cellSize = 4;  // 每个棋格子的大小 (像素)

        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int cellX = x / cellSize;
                int cellY = y / cellSize;
                bool isLight = (cellX + cellY) % 2 == 0;
                
                int index = (y * 32 + x) * 3;
                const unsigned char* color = isLight ? light : dark;
                
                data[index] = color[0];
                data[index + 1] = color[1];
                data[index + 2] = color[2];
            }
        }
        return data;
    }
}

TexturePtr PlaceHolder::Create(const string& path, const string& type) {
    auto placeholder = TexturePtr(new Texture());
    placeholder->Path = path;
    placeholder->Type = type;
    placeholder->State.store(LoadState::Placeholder);
    
    // 在主线程创建OpenGL纹理
    TaskQueue::PushTaskSync([placeholder]() {
        // 生成棋盘纹理数据
        auto data = GenerateCheckerboardData();
        
        // 创建OpenGL纹理
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        // 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        
        // 上传纹理数据
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 32, 32, 0, 
                    GL_RGB, GL_UNSIGNED_BYTE, data.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // 设置纹理属性
        placeholder->ID = textureID;
        placeholder->MarkValid(true);
        placeholder->State.store(LoadState::Ready);
    });
    
    std::lock_guard lock(s_TextureMutex);
    s_TexturePool[path] = placeholder;
    return placeholder;
}
} // namespace CubeDemo