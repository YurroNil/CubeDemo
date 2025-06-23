// include/loaders/model.h
#pragma once
#include "loaders/material.h"

using ModelLoadCallback = std::function<void()>;  // 模型加载回调

namespace CubeDemo {

// Loader:Model类
class Loaders::Model : public Loaders::Material {
private:
    ::CubeDemo::Model* m_owner;
    const string Rawpath;
    
    // 记录已打印的复用路径
    static std::unordered_set<string> m_PrintedPaths;
    static std::mutex m_PrintMutex;

    void ProcNode(aiNode* node, const aiScene* scene);
    Mesh ProcMesh(aiMesh* mesh, const aiScene* scene);

public:
    Model(const string& path, ::CubeDemo::Model* model);
    void LoadAsync(ModelLoadCallback cb); // 模型加载回调
    void LoadSync(ModelLoadCallback cb); // 同步加载模型
    void LoadModel(const string& path); // 加载模型
};
}   // namespace CubeDemo
