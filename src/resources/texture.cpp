// src/resources/texture.cpp

// 标准库
#include <iostream>
#include <future>
#include "utils/fileSystemKits.h"
// 第三方库
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "glad/glad.h"
// 项目头文件
#include "resources/texture.h"
#include "core/window.h"


namespace CubeDemo {


// 调试追踪器
std::atomic<size_t> Texture::s_TextureAliveCount{0};
TexPtrHashMap Texture::s_TexturePool;


TexturePtr Texture::Create(const string& path, const string& type) {
    // 主线程直接创建
    if (Window::IsMainThread()) {
        return CreateSync(path, type);
    }
    
    // 非主线程提交任务
    auto promise = std::make_shared<std::promise<TexturePtr>>();
    auto future = promise->get_future();
    
    Window::PushTask([=]{
        try {
            auto tex = CreateSync(path, type);
            promise->set_value(tex);
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });
    
    return future.get();
}



TexturePtr Texture::CreateSync(const string& path, const string& type) {

/* --------------调试相关-------------- */
    auto start = std::chrono::high_resolution_clock::now();

    s_TextureAliveCount++;
    std::cout << "+++ 创建纹理 [" << s_TextureAliveCount << "]: " << path;
    // GL对象创建校验
    if(!Window::IsMainThread()) {
        throw std::runtime_error("[ERROR] Texture必须在主线程创建！");
    }

/* --------------路径验证-------------- */
    // 路径验证
    if (!fs::exists(path)) {
        // 尝试在父目录中查找
        string altPath = "../" + path;
        if (!fs::exists(altPath)) {
            throw std::runtime_error("[ERROR] 纹理文件不存在: " + path);
        }
        return Create(altPath, type); // 递归尝试
    }

    // 检查纹理池
    if (auto it = s_TexturePool.find(path); it != s_TexturePool.end()) {
        if (auto tex = it->second.lock()) return tex;
    }
    // 创建新纹理
    auto tex = TexturePtr(new Texture(path, type));
    s_TexturePool[path] = tex;
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "当前进程创建的新纹理为: " << tex;
    std::cout << ", 纹理加载耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms. ";

    return tex;
}

Texture::Texture(const string& path, const string& type) 
    : Type(type), Path(path) {

    if (!Window::IsMainThread()) {
        std::cerr << "[ERROR] 纹理在非主线程创建！\n";
        std::terminate();
    }

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
        throw std::runtime_error("[ERROR] 加载纹理失败: " + path + ", 原因: " + stbi_failure_reason());
    }
    
    auto end = std::chrono::high_resolution_clock::now();

}

void Texture::Bind(unsigned int slot) const {
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, ID);
}

// 若取消此析构函数的注释，那么纹理资源会因为过早地被释放，从而导致无法被应用到网格，最后游戏中的模型全是没有纹理的黑色！！
Texture::~Texture() {
    s_TextureAliveCount--;
    std::cout << "--- 销毁纹理 [" << s_TextureAliveCount << "]: "  << Path << " (ID:" << ID << ")\n";
    // 确保在主线程释放GL资源
    if(ID != 0) { Window::PushTask([id = ID]{ glDeleteTextures(1, &id); }); }
}

}   // namespace CubeDemo
