// include/loaders/material.inl
#pragma once

namespace CubeDemo {
using TexPtrArray = std::vector<std::shared_ptr<Texture>>;

// Loaders.Material类中的模板实现

// LoaderFunc模板
template <typename LoaderFunc>
TexPtrArray Loaders::Material::LoadTextures(aiMaterial* mat, aiTextureType type, const string& type_name, LoaderFunc loader) {
    TexPtrArray textures;
    constexpr bool is_async_during_compi = std::is_invocable_v<LoaderFunc, string, string, TexLoadCallback>;

    if constexpr (is_async_during_compi) { // 异步模式分支
        auto ctx = std::make_shared<ATL::Context>();
        ctx->output = &textures;
        ctx->pendingCount.store(mat->GetTextureCount(type));

        for(unsigned i=0; i<mat->GetTextureCount(type); ++i) {
            aiString str;
            mat->GetTexture(type, i, &str);
            string path = BuildTexPath(str.C_Str());
            
            // 显式捕获path副本
            loader(path, type_name, [ctx, path](TexturePtr tex) { 
                ATL::OnTexLoaded(ctx, path, tex); 
            });
        }

        WaitForCompletion(ctx);
    } else { // 同步模式分支
        for(unsigned i=0; i<mat->GetTextureCount(type); ++i) {
            aiString str;
            mat->GetTexture(type, i, &str);
            string path = BuildTexPath(str.C_Str());
            
            if(auto tex = loader(path, type_name)) { // 直接获取返回值
                textures.push_back(tex);
            }
        }
    }
    
    return textures;
}

}   // namespace CubeDemo
