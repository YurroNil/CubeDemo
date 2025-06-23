// include/loaders/material.h
#pragma once
#include "graphics/fwd.h"
#include "loaders/async_tex.h"

namespace CubeDemo {
// 别名
using ATL = CubeDemo::Loaders::AsyncTexture;
using VertexArray = std::vector<Vertex>;
using MeshArray = std::vector<Mesh>;

class Loaders::Material {
public:

    string Directory;
    void ProcMaterial(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures, bool is_aync);

private:
    string BuildTexPath(const char* aiPath) const;
    void WaitForCompletion(const std::shared_ptr<ATL::Context>& ctx);

    template<typename LoaderFunc>
    TexPtrArray LoadTextures(aiMaterial*, aiTextureType, const string&, LoaderFunc);

    // 异步加载入口
    TexPtrArray LoadTex(aiMaterial* mat, aiTextureType type, const string& type_name, bool is_async);
};
}
// 模板实现
#include "loaders/material.inl"
