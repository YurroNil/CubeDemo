// include/loaders/material.h

#pragma once
#include "graphics/mesh.h"
#include "kits/assimp.h"
#include "loaders/asyncTex.h"

using ATL = CubeDemo::Loaders::AsyncTexture;

namespace CubeDemo {
using VertexArray = std::vector<Vertex>;
using MeshArray = std::vector<Mesh>;


class Loaders::Material {
public:
    string Directory;

    void ProcMaterial(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures, bool isAyncMode);

private:
    string BuildTexPath(const char* aiPath) const;
    void WaitForCompletion(const std::shared_ptr<ATL::Context>& ctx);

    template<typename LoaderFunc>
    TexPtrArray LoadTextures(aiMaterial*, aiTextureType, const string&, LoaderFunc);

    // 异步加载入口
    TexPtrArray LoadTex(aiMaterial* mat, aiTextureType type, const string& typeName, bool isAsync);

};

}
// 模板实现
#include "loaders/material.inl"