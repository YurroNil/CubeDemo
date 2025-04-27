// include/threads/modelLoader.h
#pragma once
#include "resources/modelData.h"
#include "threads/resourceLoader.h"

namespace CubeDemo {
using ModelLoadCallback = std::function<void()>;  // 模型加载回调

// ModelLoader类
class ModelLoader : public ModelData {
public:
    ModelLoader(const string& path);
    void LoadAsync(ModelLoadCallback cb); // 模型加载回调
    void LoadSync(ModelLoadCallback cb); // 同步加载模型（调试专用）
    bool IsReady() const;

};

}   // namespace CubeDemo
