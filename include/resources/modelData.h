// include/resources/modelData.h

#pragma once
#include <vector>
#include "graphics/mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

namespace CubeDemo {
using TexPtrArray = std::vector<TexturePtr>;
using VertexArray = std::vector<Vertex>;
using MeshArray = std::vector<Mesh>;


// 包围球结构体
struct BoundingSphere {
    vec3 Center; float Rad; // 包围球中心, 半径
    void Calc(const MeshArray& meshes); // 计算包围球
};
// 材质数据结构体
struct MaterialData {
    string Directory;
    TexPtrArray LoadTex(aiMaterial* mat, aiTextureType type, const string& typeName);
    void ProcMaterial(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures);

};

// ModelData结构体
struct ModelData : public MaterialData {
    ModelData(const string& path);

    BoundingSphere bounds; MeshArray m_meshes; const string Rawpath;
    // 异步加载状态
    std::atomic<bool> m_IsLoading{false}; std::atomic<bool> m_MeshesReady{false};

    void Draw(Shader& shader);  //  绘制模型
    void LoadModel(const string& path); // 加载模型

private:
    void ProcNode(aiNode* node, const aiScene* scene);
    Mesh ProcMesh(aiMesh* mesh, const aiScene* scene);

};


}   // namespace CubeDemo