// include/resources/materialData.h

#pragma once
#include <vector>
#include "graphics/mesh.h"
#include "utils/assimpKits.h"

namespace CubeDemo {
using TexPtrArray = std::vector<TexturePtr>;
using VertexArray = std::vector<Vertex>;
using MeshArray = std::vector<Mesh>;

// 材质数据结构体
struct MaterialData {
    string Directory;
    TexPtrArray LoadTex(aiMaterial* mat, aiTextureType type, const string& typeName);
    void ProcMaterial(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures);

    // 使用同步加载纹理（调试专用）
    TexPtrArray LoadTexSync(aiMaterial* mat, aiTextureType type, const string& typeName);
    // 使用同步处理材质（调试专用）
    void ProcMaterialSync(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures);
};

}