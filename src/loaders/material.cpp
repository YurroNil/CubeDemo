// src/loaders/material.cpp

// 标准库
#include <iostream>
#include "kits/file_system.h"
// 项目头文件
#include "loaders/material.h"
#include "threads/taskQueue.h"


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

// 异步
void MaL::ProcMaterial(aiMesh*& mesh, const aiScene*& scene, TexPtrArray& textures, bool isAyncMode) {

    // 同步加载模式的材质索引检查
    if (isAyncMode && mesh->mMaterialIndex < 0 || mesh->mMaterialIndex >= scene->mNumMaterials) {
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

    for (const auto& [primaryType, fallbackType, typeName] : strategies) {
        auto maps = LoadTex(material, primaryType, typeName, isAyncMode);
        if (maps.empty() && fallbackType != aiTextureType_NONE) {
            maps = LoadTex(material, fallbackType, typeName, isAyncMode);
        }
        textures.insert(textures.end(), maps.begin(), maps.end());
    }
}

TexPtrArray MaL::LoadTex(aiMaterial* mat, aiTextureType type, const string& typeName, bool isAsync) {
    if (isAsync) {
        return LoadTextures(mat, type, typeName,
            [](const string& path, const string& type, auto&& cb) { // 通用回调
                TL::LoadAsync(path, type, std::forward<decltype(cb)>(cb));
            }
        );
    } else {
        return LoadTextures(mat, type, typeName,
            [](const string& path, const string& type) {
                return TL::LoadSync(path, type);
            }
        );
    }
}

}