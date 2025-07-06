// include/loaders/progress_tracker.h
#pragma once

namespace CubeDemo::Loaders {

class ProgressTracker {
public:
    static ProgressTracker& Get();
    
    // 资源类型枚举
    enum ResourceType {
        MODEL_FILE,
        MODEL_GEOMETRY,
        TEXTURE_IO,
        TEXTURE_GPU,
        FONT
    };
    
    // 添加资源到跟踪系统
    void AddResource(ResourceType type, const string& path, size_t weight = 0);
    
    // 更新资源状态
    void UpdateProgress(ResourceType type, const string& path, float progress);
    void FinishResource(ResourceType type, const string& path);
    
    // 获取总进度 (0.0 - 1.0)
    float GetOverallProgress() const;
    
    // 注册进度回调
    void SetProgressCallback(std::function<void(float)> callback);
    
    // 重置跟踪器
    void Reset();
    std::mutex& GetMutex() { return m_Mutex; }

private:
    ProgressTracker() = default;
    
    struct Resource {
        size_t weight;
        float progress = 0.0f;
        bool completed = false;
    };
    
    std::atomic<size_t> m_TotalWeight{0};
    std::atomic<size_t> m_CompletedWeight{0};
    std::unordered_map<ResourceType, std::unordered_map<string, Resource>> m_Resources;
    mutable std::mutex m_Mutex;
    std::function<void(float)> m_ProgressCB;
};
}   // namespace CubeDemo::Loaders
