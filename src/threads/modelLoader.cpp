// src/threads/modelLoader.cpp
#include "threads/modelLoader.h"
#include "threads/textureLoader.h"
// 标准库
#include "utils/streams.h"
#include "utils/fileSystemKits.h"


namespace CubeDemo {

// MeterialData结构体方法实现
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
            TaskQueue::ProcTasks(); // 处理主线程任务
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    return textures;
}


// ModelLoader类实现

// 构造函数ModelLoader作为中间层，需要将路径传递给ModelData类
ModelLoader::ModelLoader(const string& path) : ModelData(path) {}

// 异步加载模型
void ModelLoader::LoadAsync(ModelLoadCallback cb) {
    m_IsLoading = true;
    ResourceLoader::EnqueueIOJob([this, cb]{

        std::cout << "\n---[ModelAsyncLoader] 开始加载模型..." << std::endl;
        this->LoadModel(Rawpath);
        std::cout << "---[ModelAsyncLoader] 加载模型结束" << std::endl;
        
        TaskQueue::AddTasks([this, cb]{
            // 通知所有网格更新纹理引用
            ModelData::m_MeshesReady.store(true, std::memory_order_release);
            for(auto& mesh : m_meshes) {
                mesh.UpdateTextures(mesh.m_textures); 
            }
            m_IsLoading = false;
            cb();
        }, true);
    });
}

bool ModelLoader::IsReady() const { return ModelData::m_MeshesReady.load(std::memory_order_acquire); }


} // namespace CubeDemo