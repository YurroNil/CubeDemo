// include/loaders/texture.h
#pragma once
#include "loaders/fwd.h"
#include "resources/texture.h"

namespace CubeDemo {

// 乱七八糟的别名
using millisec = std::chrono::milliseconds;
using ImagePtr = std::shared_ptr<Loaders::Image>;
using TexLoadCallback = std::function<void(TexturePtr)>; // 纹理加载回调
using TL = Loaders::Texture;

// loaders.texture类继承texture类

class Loaders::Texture : public CubeDemo::Texture {
public:

    // 异步加载纹理
    static void LoadAsync(const string& path, const string& type, TexLoadCallback cb);
    // 同步 加载纹理
    static TexturePtr LoadSync(const string& path, const string& type);
    static void CreateTexAsync(const string& path, const string& type, TexLoadCallback cb);
    static void ClearCache();


    //------------------------ 统计信息 ------------------------//
    inline static std::atomic<int32_t> s_TextureAliveCount{0};  // 存活纹理计数
    inline static std::atomic<uint32_t> s_ActiveLoads{0};       // 进行中的
    //------------------------ 静态资源 ------------------------//
    static TexPtrHashMap s_TexturePool;   // 纹理资源池
    static std::mutex s_TextureMutex;     // 资源池互斥锁
    static TexturePtr CreateFromData(ImagePtr data, const string& path, const string& type);

private:
    //------------------------ 私有实现 ------------------------//
    
    // 核心私有方法
    static TexturePtr TryGetCached(const string& path);

    // 重试系统相关
    struct RetrySystem {
        inline static std::unordered_map<string, int> Counters; ///< 路径重试计数器
        inline static std::mutex Mutex;                              ///< 重试系统互斥锁
        inline static constexpr int MAX_RETRY = 3;                   ///< 最大重试次数
        inline static constexpr auto BASE_DELAY = millisec(100); ///< 基础延迟
    };
    static string GetStatePrint(TexturePtr tex);

    // 记录已打印的复用路径
    static std::unordered_set<string> m_PrintedPaths;
    static std::mutex m_PrintMutex;
};
}   // namespace CubeDemo
