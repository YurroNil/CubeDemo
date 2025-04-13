// src/resources/texture.cpp

#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "resources/texture.h"
#include "glad/glad.h"
#include <filesystem>
namespace fs = std::filesystem;

namespace CubeDemo {

std::unordered_map<string, std::weak_ptr<Texture>> Texture::s_TexturePool;


std::shared_ptr<Texture> Texture::Create(const string& path, const string& type) {
    std::cout << "当前检查的纹理路径是: " << path << std::endl;
    // 更严格的路径验证
    if (!fs::exists(path)) {
        // 尝试在父目录中查找
        std::string altPath = "../" + path;
        if (!fs::exists(altPath)) {
            throw std::runtime_error("纹理文件不存在: " + path);
        }
        return Create(altPath, type); // 递归尝试
    }

    // 检查纹理池
    if (auto it = s_TexturePool.find(path); it != s_TexturePool.end()) {
        if (auto tex = it->second.lock()) return tex;
    }

    // 创建新纹理
    auto tex = std::shared_ptr<Texture>(new Texture(path, type));
    s_TexturePool[path] = tex;
    
    std::cout << "当前进程创建的新纹理为: " << tex << std::endl;
    return tex;
}

Texture::Texture(const string& path, const string& type) 
    : Type(type), Path(path) 
{
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_2D, ID);
    
    int width, height, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    
    if (data) {
        GLenum format = GL_RGBA;
        switch (channels) {
            case 1: format = GL_RED; break;
            case 3: format = GL_RGB; break;
            case 4: format = GL_RGBA; break;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        stbi_image_free(data);
    } else {
        glDeleteTextures(1, &ID);
        throw std::runtime_error("加载纹理失败: " + path);
    }
}

void Texture::Bind(unsigned int slot) const {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, ID);
}

}