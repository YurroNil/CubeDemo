// include/loaders/model.h
#pragma once
#include "graphics/boundingSphere.h"
#include "loaders/resource.h"

using ModelLoadCallback = std::function<void()>;  // 模型加载回调

namespace CubeDemo {
// Loader.Model类
class Loaders::Model : public Loaders::Material {
public:
    BoundingSphere bounds; MeshArray m_meshes;
    const string Rawpath;

    // 异步加载状态
    std::atomic<bool> m_IsLoading = false;
    std::atomic<bool> m_MeshesReady = false;

    void Draw(Shader& shader);  //  绘制模型

    mat4 m_ModelMatrix = mat4(1.0f);
    void ProcNode(aiNode* node, const aiScene* scene);
    Mesh ProcMesh(aiMesh* mesh, const aiScene* scene);

    Model(const string& path);
    void LoadAsync(ModelLoadCallback cb); // 模型加载回调
    void LoadSync(ModelLoadCallback cb); // 同步加载模型（调试专用）
    bool IsReady() const;
    void LoadModel(const string& path); // 加载模型

};
}   // namespace CubeDemo
