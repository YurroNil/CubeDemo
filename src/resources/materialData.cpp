// src/resources/materialData.cpp

// 标准库
#include <iostream>
#include "utils/fileSystemKits.h"
// 项目头文件
#include "resources/materialData.h"
#include "threads/taskQueue.h"
#include "resources/placeHolder.h"

namespace CubeDemo {

// 加载纹理（异步）
TexPtrArray MaterialData::LoadTex(aiMaterial* mat, aiTextureType type, const string& typeName) {

    TexPtrArray textures;
    const unsigned textureCount = mat->GetTextureCount(type);
    
    struct AsyncContext {
        std::mutex mutex;
        std::vector<TexturePtr> loadedTextures;
        std::atomic<int> pendingCount{0};
        std::promise<void> completionPromise;
        TexPtrArray* outputTextures;
    };
    
    auto context = std::make_shared<AsyncContext>();
    context->outputTextures = &textures;
    context->pendingCount.store(textureCount);

    std::cout << "[断点D]" << std::endl;

    for(unsigned i=0; i<textureCount; ++i) {
        aiString str;
        mat->GetTexture(type, i, &str);
        
        string path = this->Directory + "/textures/" + fs::path(str.C_Str()).filename().string();

        std::cout << "[断点E]" << std::endl;

        TextureLoader::LoadAsync(path, typeName, [context, path](TexturePtr tex) {

            std::cout << "[断点J]" << std::endl;
            std::lock_guard<std::mutex> lock(context->mutex);

            // 处理各种加载状态
            if (tex) {
                switch(tex->State.load()) {
                case Texture::LoadState::Ready:
                    context->loadedTextures.push_back(tex);
                    std::cout << "[Success] 加载完成: " << path << "\n";
                    break;
                case Texture::LoadState::Placeholder:
                    context->loadedTextures.push_back(tex);
                    std::cout << "[Warning] 使用占位纹理: " << path << "\n";
                    break;
                case Texture::LoadState::Failed:
                    std::cerr << "[Error] 最终加载失败: " << path << "\n";
                    break;
                default: 
                    break;
                }
            }
            
            // 原子递减并检查是否是最后一个任务
            if(context->pendingCount.fetch_sub(1) == 1) {
                *context->outputTextures = std::move(context->loadedTextures);
                context->completionPromise.set_value();
            }
            std::cout << "[断点Z]" << std::endl;

        });
    }
    // 等待逻辑
    if (textureCount > 0) {
        auto future = context->completionPromise.get_future();
        while (future.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
            int processed = 0;
            TaskQueue::ProcTasks(processed); // 处理主线程任务
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return textures;
}

// 处理材质（异步）
void MaterialData::ProcMaterial(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures) {

    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    std::cout << "[断点C]" << std::endl;
    // 漫反射贴图
    auto diffuseMaps = LoadTex(material, aiTextureType_DIFFUSE, "texture_diffuse");

    if (diffuseMaps.empty()) { // 如果未找到，尝试其他类型
        diffuseMaps = LoadTex(material, aiTextureType_BASE_COLOR, "texture_diffuse");
    }  
    textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());


    // 加载反射贴图
    auto reflectionMaps = LoadTex(material, aiTextureType_REFLECTION, "texture_reflection");
    textures.insert(textures.end(), reflectionMaps.begin(), reflectionMaps.end());

    // 法线贴图（map_Bump）
    auto normalMaps = LoadTex(material, aiTextureType_NORMALS, "texture_normal");
    textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

    // 高光贴图（map_Ns）
    auto specularMaps = LoadTex(material, aiTextureType_SPECULAR, "texture_specular");
    if (specularMaps.empty()) { // 如果未找到，尝试其他类型
        specularMaps = LoadTex(material, aiTextureType_SHININESS, "texture_diffuse");
    }
    textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

    // 环境光遮蔽（map_Ka）
    auto aoMaps = LoadTex(material, aiTextureType_AMBIENT, "texture_ao");

    textures.insert(textures.end(), aoMaps.begin(), aoMaps.end());

    std::cout << "[断点END]" << std::endl;
}


/* ---------调试专用--------- */

// 加载纹理（同步）
TexPtrArray MaterialData::LoadTexSync(aiMaterial* mat, aiTextureType type, const string& typeName) {

    TexPtrArray textures;
    const unsigned textureCount = mat->GetTextureCount(type);

    for(unsigned i=0; i<textureCount; ++i) {
        aiString str;
        mat->GetTexture(type, i, &str);

        string path = this->Directory + "/textures/" + fs::path(str.C_Str()).filename().string();

        if (!fs::exists(path)) {
            string altPath = "../" + path;
            if (fs::exists(altPath)) {
                path = altPath;
            } else {
                throw std::runtime_error("纹理文件不存在: " + path);
            }
        }

        try {
        TexturePtr tex = TextureLoader::LoadSync(path, typeName);
        textures.push_back(tex);

        } catch (const std::exception& e) {
        std::cerr << "加载失败: " << e.what() << std::endl;
        // 使用占位纹理
        textures.push_back(PlaceHolder::Create(path, typeName));
        }

        std::cout << "[MaterialData:LoadTexSync] 加载结束: " << path << std::endl;

        // 打印调试信息
        std::cout << "加载纹理路径: " << path << " | 文件存在: " << (fs::exists(path) ? "是" : "否") << std::endl;
    }

    return textures;
}

// 处理材质（同步）
void MaterialData::ProcMaterialSync(aiMesh* &mesh, const aiScene* &scene, TexPtrArray& textures) {
    // 处理材质
    if(mesh->mMaterialIndex >= scene->mNumMaterials) {
        std::cerr << "无效的材质索引: " << mesh->mMaterialIndex << "/" << scene->mNumMaterials << std::endl;
        return;
    }

     if (mesh->mMaterialIndex >= 0) {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        // 漫反射贴图
        auto diffuseMaps = LoadTexSync(material, aiTextureType_DIFFUSE, "texture_diffuse");
        std::cout << "加载漫反射贴图数量: " << diffuseMaps.size() << std::endl;
        if (diffuseMaps.empty()) { // 如果未找到，尝试其他类型
            diffuseMaps = LoadTexSync(material, aiTextureType_BASE_COLOR, "texture_diffuse");
        }
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

         // 加载反射贴图
        auto reflectionMaps = LoadTexSync(material, aiTextureType_REFLECTION, "texture_reflection");
        std::cout << "加载反射贴图数量: " << reflectionMaps.size() << std::endl;
        textures.insert(textures.end(), reflectionMaps.begin(), reflectionMaps.end());

        // 法线贴图（map_Bump）
        auto normalMaps = LoadTexSync(material, aiTextureType_NORMALS, "texture_normal");
        std::cout << "加载法线贴图数量: " << normalMaps.size() << std::endl;
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

        // 高光贴图（map_Ns）
        auto specularMaps = LoadTexSync(material, aiTextureType_SPECULAR, "texture_specular");
        std::cout << "加载高光贴图数量: " << specularMaps.size() << std::endl;
        if (specularMaps.empty()) { // 如果未找到，尝试其他类型
            specularMaps = LoadTexSync(material, aiTextureType_SHININESS, "texture_diffuse");
        }
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

        // 环境光遮蔽（map_Ka）
        auto aoMaps = LoadTexSync(material, aiTextureType_AMBIENT, "texture_ao");
        std::cout << "加载环境光遮蔽贴图数量: " << aoMaps.size() << std::endl;
        textures.insert(textures.end(), aoMaps.begin(), aoMaps.end());
    }
}
}