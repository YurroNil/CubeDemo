// src/loaders/material.cpp
#include "pch.h"
#include "loaders/material.h"
#include "threads/taskQueue.h"
#include "loaders/texture.h"

// 别名
using MaL = CubeDemo::Loaders::Material;

namespace CubeDemo {

string MaL::BuildTexPath(const char* aiPath) const {
    return this->Directory + "/textures/" + fs::path(aiPath).filename().string();
}

void MaL::WaitForCompletion(const std::shared_ptr<ATL::Context>& ctx) {
    auto future = ctx->completionPromise.get_future();
    while (future.wait_for(millisec(1)) != std::future_status::ready) {
        int processed = 0;
        TaskQueue::ProcTasks(processed);
        std::this_thread::sleep_for(millisec(1));
    }
}
void MaL::ProcMaterial(aiMesh*& mesh, const aiScene*& scene, TexPtrArray& textures, bool is_aync) {

    // 同步加载模式的材质索引检查
    if (is_aync && mesh->mMaterialIndex < 0 || mesh->mMaterialIndex >= scene->mNumMaterials) {
        std::cerr << "无效的材质索引: " << mesh->mMaterialIndex << "\n";
        return;
    }

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    
    // 加载策略：类型 + 后备类型
    const std::vector<std::tuple<aiTextureType, aiTextureType, string>> strategies = {
        {aiTextureType_DIFFUSE, aiTextureType_BASE_COLOR, "texture_diffuse"},
        {aiTextureType_REFLECTION, aiTextureType_NONE, "texture_reflection"},
        {aiTextureType_NORMALS, aiTextureType_NONE, "texture_normal"},
        {aiTextureType_SPECULAR, aiTextureType_SHININESS, "texture_specular"},
        {aiTextureType_AMBIENT, aiTextureType_NONE, "texture_ao"}
    };

    for (const auto& [primaryType, fallbackType, type_name] : strategies) {
        auto maps = LoadTex(material, primaryType, type_name, is_aync);
        if (maps.empty() && fallbackType != aiTextureType_NONE) {
            maps = LoadTex(material, fallbackType, type_name, is_aync);
        }
        textures.insert(textures.end(), maps.begin(), maps.end());
    }
}

TexPtrArray MaL::LoadTex(aiMaterial* mat, aiTextureType type, const string& type_name, bool is_async) {
    if (is_async) {
        return LoadTextures(mat, type, type_name,
            [](const string& path, const string& type, auto&& cb) { // 通用回调
                TL::LoadAsync(path, type, std::forward<decltype(cb)>(cb));
            }
        );
    } else {
        return LoadTextures(mat, type, type_name,
            [](const string& path, const string& type) {
                return TL::LoadSync(path, type);
            }
        );
    }
}

}