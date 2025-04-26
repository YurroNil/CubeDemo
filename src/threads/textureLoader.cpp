// src/threads/textureLoader.cpp

// 项目头文件
#include "resources/imageData.h"
// 标准库
#include "utils/fileSystemKits.h"

namespace CubeDemo {
extern void TrackGLObj(GLuint id, const string& type);
extern void UntrackGLObj(GLuint id);


// 静态成员初始化
TexPtrHashMap TextureLoader::s_TexturePool;
std::mutex TextureLoader::s_TextureMutex;


TexturePtr TextureLoader::Create(const string& path, const string& type) {
    // 第一层无锁检查
    if(auto cached = TryGetCached(path); cached) {
        return cached;
    }

    std::lock_guard lock(s_TextureMutex);
    if (auto it = s_TexturePool.find(path); it != s_TexturePool.end()) {
        if (auto tex = it->second.lock()) {
            return tex->m_Valid.load() ? tex : nullptr;
        }
    }

    // 创建占位符
    auto placeholder = Create(path, type);
    s_TexturePool[path] = placeholder;
    // 带重试机制的异步加载
    PlaceHolder::ScheAsyncLoad(path, type, placeholder);

    return placeholder;
}

TexturePtr TextureLoader::TryGetCached(const string& path) {
    std::lock_guard lock(s_TextureMutex);
    if(auto it = s_TexturePool.find(path); it != s_TexturePool.end()) {
        if(auto tex = it->second.lock()) {
            return tex->m_Valid.load() ? tex : nullptr;
        }
    }
    return nullptr;
}

// 异步加载纹理
void TextureLoader::LoadAsync(const string& path, const string& type, TexLoadCallback cb) {

    s_ActiveLoads.fetch_add(1);
    auto imageData = ImageData::Load(path); // 返回shared_ptr

    std::cout << "[断点F]" << std::endl;

    ResourceLoader::EnqueueIOJob([=]() mutable {
        std::cout << "[断点I]" << std::endl;

        // 将GL操作提交到主线程队列（非阻塞）
        TaskQueue::AddTasks([imageData, path, type, cb]() {
            auto tex = CreateFromData(imageData, path, type);
            cb(tex);
        }, true);
    });
}

TexturePtr TextureLoader::CreateFromData(ImageDataPtr data, const string& path, const string& type) {
    // 参数验证
    if(!data || !data->data) { throw std::runtime_error("无效的纹理数据: " + path); }

    auto tex = TexturePtr(new Texture());
    tex->Path = path;
    tex->Type = type;

    GLuint tempID;
    glGenTextures(1, &tempID);
    tex->ID.store(tempID); // 使用原子存储
    TrackGLObj(tex->ID, "Texture");
    glBindTexture(GL_TEXTURE_2D, tempID);

    GLenum format = GL_RGBA;
    switch(data->channels) {
        case 1: format = GL_RED;  break;
        case 3: format = GL_RGB;  break;
        case 4: format = GL_RGBA; break;
        default: throw std::runtime_error("不支持的通道数: " + std::to_string(data->channels));
    }

    glTexImage2D(GL_TEXTURE_2D, 0, format, data->width, data->height, 0, format, GL_UNSIGNED_BYTE, data->data.get());
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glGenerateMipmap(GL_TEXTURE_2D);

    std::lock_guard lock(s_TextureMutex);
    s_TexturePool[path] = tex;
    tex->State.store(LoadState::Ready);
    return tex;
}

// 暂无使用. 作为备份/进度回溯使用
TexturePtr TextureLoader::CreateSync(const string& path, const string& type) {

/* --------------调试相关-------------- */
    auto start = std::chrono::high_resolution_clock::now();

    s_TextureAliveCount++;
    std::cout << "+++ 创建纹理 [" << s_TextureAliveCount << "]: " << path;
    // GL对象创建校验
    if(!TaskQueue::IsMainThread()) {
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
    auto tex = TexturePtr(new Texture());
    s_TexturePool[path] = tex;
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "当前进程创建的新纹理为: " << tex;
    std::cout << ", 纹理加载耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms. ";

    return tex;
}

}   // namespace CubeDemo