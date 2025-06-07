// src/loaders/texture.cpp
#include "pch.h"
#include "resources/placeHolder.h"
#include "threads/diagnostic.h"
#include "loaders/texture.h"
#include "resources/texture.h"

// 别名
using TLS = CubeDemo::Texture::LoadState;

namespace CubeDemo {

// 静态成员初始化
TexPtrHashMap TL::s_TexturePool;
std::mutex TL::s_TextureMutex;
std::unordered_set<string> TL::s_PrintedPaths;
std::mutex TL::s_PrintMutex; 


TexturePtr TL::TryGetCached(const string& path) {
    std::lock_guard lock(s_TextureMutex);
    auto it = s_TexturePool.find(path);
    if (it == s_TexturePool.end()) return nullptr;
    
    auto tex = it->second.lock();
    if (!tex || !tex->m_Valid.load()) return nullptr;
    
    return tex->State.load() == TLS::Ready ? tex : nullptr;
}

// 异步加载纹理
void TL::LoadAsync(const string& path, const string& type, TexLoadCallback cb) {

    // 优先返回缓存
    if (auto cached = TryGetCached(path)) {
        TaskQueue::AddTasks([cached, cb] { cb(cached); }, true);
        return;
    }

    auto& diag = Diagnostic::Get();
    std::cout << "[TEXTURE] 提交异步加载 路径:" << path << " 提交线程:" << std::this_thread::get_id() << "\n";

    s_ActiveLoads.fetch_add(1);

    // 记录加载状态, 标记为进行中
    diag.stats.texturesLoaded--;

    RL::EnqueueIOJob([=]() mutable {

        try {
            CreateTexAsync(path, type, cb);

        } catch(...) {
             // 纹理创建失败时保留占位符
            auto placeholder = PlaceHolder::Create(path, type);
            placeholder->State.store(TLS::Failed);
            cb(placeholder);

            std::cerr << "[TEXTURE] IO加载失败 路径:" << path << "\n";
        }
    });
}

// 异步创建纹理
void TL::CreateTexAsync(const string& path, const string& type, TexLoadCallback cb) {
    auto image_data = IL::Load(path); // 返回shared_ptr

    std::cout << "[TEXTURE] 开始IO加载 路径:" << path << " 处理线程:" << std::this_thread::get_id() << "\n";

    // 将GL操作提交到主线程队列（非阻塞）
    TaskQueue::AddTasks([image_data, path, type, cb]() {

        // 再次检查缓存
        std::lock_guard lock(s_TextureMutex);
        auto existing = s_TexturePool[path].lock();
        if (existing && existing->State == TLS::Ready) {
            {
                std::lock_guard lock(s_PrintMutex);
                if (s_PrintedPaths.insert(path).second) {
                    std::cout << "[优化] 异步复用: " << path << std::endl;
                }
            }
            cb(existing);
            return;
        }

        // 创建纹理
        auto tex = CreateFromData(image_data, path, type);
        
        if (existing) {
            // 转移OpenGL资源到占位符
            existing->ID.store(tex->ID.load());
            existing->State.store(TLS::Ready);
            cb(existing);
        } else {
            s_TexturePool[path] = tex;
            cb(tex);
        }

        std::cout << "[TEXTURE] 完成加载回调 路径:" << path << " 状态:" << static_cast<int>(tex->State.load()) << "\n";
        
        cb(tex);
    }, true);
}

// 同步加载纹理
TexturePtr TL::LoadSync(const string& path, const string& type) {

    // 第一层带锁缓存检查
    while(true) {
        std::lock_guard lock(s_TextureMutex);

        // 在纹理池中查找指定路径path对应的条目, 并返回一个迭代器给it
        auto it = s_TexturePool.find(path);

        // 若返回的迭代器(即it)存在的话, 就不break
        if (it == s_TexturePool.end()) break;

        // 尝试查找哈希表中存储的weak_ptr<Texture>, 然后试图升级为shared_ptr
        auto tex = it->second.lock();
        // 失败则退出
        if (tex == nullptr) return nullptr;

        // 如果tex不是nullptr那么将执行复用操作
        {
            std::lock_guard lock(s_PrintMutex);
            if (s_PrintedPaths.insert(path).second) {
                std::cout << "[TL:loadSync] 正在检查路径: " << path <<", 纹理状态: " << GetStatePrint(tex) << std::endl;
            }
        }

        // 等待异步加载完成（包括占位符转换）
        while (tex->State == TLS::Loading || tex->State == TLS::Placeholder) std::this_thread::sleep_for(millisec(1));

        if (tex->State != TLS::Ready) break;

        {
            std::lock_guard lock(s_PrintMutex);
            if (s_PrintedPaths.insert(path).second) { // 如果首次插入成功
                std::cout << "[优化] 同步复用: " << path << std::endl;
            }
        }
        return tex;
    }

    // 同步加载流程
    try {
        // 加载图像数据
        auto image_data = Image::Load(path);
        
        // 在主线程执行OpenGL操作
        TexturePtr tex;
        if(TaskQueue::IsMainThread()) {
            tex = CreateFromData(image_data, path, type);
        } else {
            tex = TaskQueue::PushTaskSync([&]{
                return CreateFromData(image_data, path, type);
            });
        }

        // 更新状态
        std::lock_guard lock(s_TextureMutex);
        s_TexturePool[path] = tex;
        tex->State.store(TLS::Ready);
         // 添加状态验证
        if (tex->State != TLS::Ready) {
            throw std::runtime_error("纹理最终状态异常: " + path);
        }
        return tex;
    } 
    catch(const std::exception& e) {
        // 失败处理
        std::cerr << "[SyncLoad] 加载失败 " << path << ": " << e.what() << "\n";
        // 创建占位符
        auto placeholder = PlaceHolder::Create(path, type);
        placeholder->State.store(TLS::Failed);
        return nullptr;
    }
}


TexturePtr TL::CreateFromData(ImagePtr data, const string& path, const string& type) {
    auto& diag = Diagnostic::Get();
    // 确保在主线程创建
    if(!TaskQueue::IsMainThread()) throw std::runtime_error("OpenGL资源必须在主线程创建!");
    // 参数验证
    if(!data || !data->data) throw std::runtime_error("无效的纹理数据: " + path);

try {
    // 记录创建过程
    std::cout << "[TEXTURE] 开始创建GL纹理 路径:" << path << " 尺寸:" << data->width << "x" << data->height << "\n";

    auto tex = TexturePtr(new Texture());
    tex->Path = path;
    tex->Type = type;

    GLuint temp_id;
    glGenTextures(1, &temp_id);
    tex->ID.store(temp_id); // 使用原子存储
    glBindTexture(GL_TEXTURE_2D, temp_id);
    
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
    tex->State.store(TLS::Ready);

    return tex;

} catch(...) {
    std::cerr << "[TEXTURE] 创建失败 路径:" << path << "\n";
    throw;
}

}

string TL::GetStatePrint(TexturePtr tex) {
    string name;

    if (tex->State == LoadState::Uninitialized) return "LoadState: Uninited";
    if (tex->State == LoadState::Placeholder) return "LoadState: PlaceHolder";
    if (tex->State == LoadState::Loading) return "LoadState: Loading";
    if (tex->State == LoadState::Init) return "LoadState: Init";
    if (tex->State == LoadState::Ready) return "LoadState: Ready";
    if (tex->State == LoadState::Failed) return "LoadState: Failed";
    return "LoadState: ??";
}

}   // namespace CubeDemo