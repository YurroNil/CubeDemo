// include/threads/textureLoader.h
#pragma once

// 项目头文件
#include "resources/texture.h"
// 标准库
#include <functional>
#include <format>

namespace CubeDemo {

struct ImageData;
using ImageDataPtr = std::shared_ptr<ImageData>;
using TexLoadCallback = std::function<void(TexturePtr)>; // 纹理加载回调

// TextureLoader类
class TextureLoader : public Texture {
public:

    //------------------------ 核心接口 ------------------------//

    // 创建+同步加载纹理（调试专用）
    static TexturePtr CreateSync(const string& path, const string& type);
    // 仅创建纹理（异步）
    static TexturePtr Create(const string& path, const string& type);
    // 仅加载纹理（异步）
    static void LoadAsync(const string& path, const string& type, TexLoadCallback cb);
    // 仅同步加载纹理（调试专用）
    static TexturePtr LoadSync(const string& path, const string& type);

    //------------------------ 统计信息 ------------------------//
    inline static std::atomic<int32_t> s_TextureAliveCount{0};  // 存活纹理计数
    inline static std::atomic<uint32_t> s_ActiveLoads{0};       // 进行中的
    //------------------------ 静态资源 ------------------------//
    static TexPtrHashMap s_TexturePool;   // 纹理资源池
    static std::mutex s_TextureMutex;     // 资源池互斥锁

    static TexturePtr CreateFromData(ImageDataPtr data, const string& path, const string& type);


private:
    //------------------------ 私有实现 ------------------------//
    
    // 核心私有方法

    static TexturePtr TryGetCached(const string& path);

    // 重试系统相关
    struct RetrySystem {
        inline static std::unordered_map<string, int> Counters; ///< 路径重试计数器
        inline static std::mutex Mutex;                              ///< 重试系统互斥锁
        inline static constexpr int MAX_RETRY = 3;                   ///< 最大重试次数
        inline static constexpr auto BASE_DELAY = std::chrono::milliseconds(100); ///< 基础延迟
    };

};


}   // namespace CubeDemo

