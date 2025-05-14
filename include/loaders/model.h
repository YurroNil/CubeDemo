// include/loaders/model.h
#pragma once
#include "graphics/boundingSphere.h"
#include "loaders/resource.h"
#include "graphics/lod.h"

using ModelLoadCallback = std::function<void()>;  // 模型加载回调

namespace CubeDemo {

// Loader:Model类
class Loaders::Model : public Loaders::Material {
private:

    Graphics::LODSystem m_LODSystem; // 新增LOD系统成员

    MeshArray m_meshes;
    MeshArray m_meshes_copy;
    const string Rawpath;

    // 异步加载状态
    std::atomic<bool> m_IsLoading = false;
    std::atomic<bool> m_MeshesReady = false;


    mat4 m_ModelMatrix = mat4(1.0f);

    void ProcNode(aiNode* node, const aiScene* scene);
    Mesh ProcMesh(aiMesh* mesh, const aiScene* scene);

public:
    BoundingSphere bounds;

    Model(const string& path);
    void LoadAsync(ModelLoadCallback cb); // 模型加载回调
    void LoadSync(ModelLoadCallback cb); // 同步加载模型
    bool IsReady() const;
    void LoadModel(const string& path); // 加载模型

    // 乱七八糟的Getters
    const Graphics::LODSystem& GetLODSystem() const;
    const std::atomic<bool>& isLoading() const;
    const MeshArray& GetMeshes() const;
    const mat4& GetModelMatrix() const;

};
}   // namespace CubeDemo
