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
    bool IsReady() const;

};

}   // namespace CubeDemo
