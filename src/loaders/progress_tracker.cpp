// src/progress/progress_tracker.cpp
#include "pch.h"
#include "loaders/progress_tracker.h"
#include "ui/screens/loading.h"

namespace CubeDemo::Loaders {
using Tracker = ProgressTracker;

Tracker& Tracker::Get() {
    static Tracker instance;
    return instance;
}

void Tracker::AddResource(ResourceType type, const string& path, size_t weight) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    // 动态权重调整 - 根据资源类型和路径
    if (weight == 0) {
        // 根据文件扩展名调整权重
        size_t baseWeight = 10;
        if (
            path.find(".obj") != string::npos ||
            path.find(".fbx") != string::npos ||
            path.find(".dae") != string::npos ||
            path.find(".gltf")!= string::npos ||
            path.find(".glb") != string::npos
        ) baseWeight = 30;
        else if (path.find(".png") != string::npos) baseWeight = 5;
        else if (path.find(".jpg") != string::npos) baseWeight = 8;
        
        const std::unordered_map<ResourceType, size_t> TYPE_MULTIPLIERS = {
            {MODEL_FILE, 3},
            {MODEL_GEOMETRY, 5},
            {TEXTURE_IO, 1},
            {TEXTURE_GPU, 2},
            {FONT, 1}
        };
        
        weight = baseWeight * TYPE_MULTIPLIERS.at(type);
    }
    
    // 确保类型映射存在
    auto& typeMap = m_Resources[type];
    if (typeMap.find(path) == typeMap.end()) {
        typeMap[path] = {weight, 0.0f, false};
        m_TotalWeight += weight;
        
        // 调试日志
        // std::cout << "[加载进度] 添加资源: " << path << " 任务类型: " << type << " 权重值: " << weight << " 共计: " << m_TotalWeight << std::endl;
    }
}

void Tracker::UpdateProgress(ResourceType type, const string& path, float progress) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    // 确保资源存在
    auto typeIt = m_Resources.find(type);

    // 如果资源类型或路径不存在，则返回
    if (typeIt == m_Resources.end()) return;

    // 如果资源路径不存在，则返回
    auto resIt = typeIt->second.find(path);
    if (resIt == typeIt->second.end()) return;

    // 获取资源
    Resource& res = resIt->second;
    
    // 计算权重增量（带平滑过渡）
    float effectiveProgress = std::clamp(progress, 0.0f, 1.0f);
    float delta = (effectiveProgress - res.progress) * res.weight;
    
    // 更新进度
    res.progress = effectiveProgress;
    m_CompletedWeight += static_cast<size_t>(delta);
    
    // 调试输出
    // std::cout << "[加载进度] 进度已经更新. 当前处理的资源路径:" << path << " 任务类型: " << type << " 当前任务的进度: " << effectiveProgress * 100 << "%" << " 总体进度: " << GetOverallProgress() * 100 << "%" << std::endl;

    // 触发回调
    if (m_ProgressCB) m_ProgressCB(GetOverallProgress());
}

void Tracker::FinishResource(ResourceType type, const string& path) {
    UpdateProgress(type, path, 1.0f);
    
    std::lock_guard<std::mutex> lock(m_Mutex);

    auto typeIt = m_Resources.find(type);
    if (typeIt == m_Resources.end()) return;

    auto resIt = typeIt->second.find(path);
    if (resIt == typeIt->second.end()) return;

    resIt->second.completed = true;
}

float Tracker::GetOverallProgress() const {
    size_t completed = m_CompletedWeight.load(std::memory_order_relaxed);
    size_t total = m_TotalWeight.load(std::memory_order_relaxed);
    
    if (total == 0) return 0.0f;
    return static_cast<float>(completed) / total;
}

void Tracker::SetProgressCallback(std::function<void(float)> callback) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_ProgressCB = callback;
}
void Tracker::Reset() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Resources.clear();
    m_TotalWeight = 0;
    m_CompletedWeight = 0;
}
} // namespace CubeDemo::Loaders
