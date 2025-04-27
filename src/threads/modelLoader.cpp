// src/threads/modelLoader.cpp
#include "threads/modelLoader.h"
// 标准库
#include "utils/streams.h"

namespace CubeDemo {

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

// 同步加载模型（调试专用）
void ModelLoader::LoadSync(ModelLoadCallback cb) {
    m_IsLoading = true;
    std::cout << "\n---[ModelLoader] 使用同步加载模式加载模型..." << std::endl;

    this->LoadModel(Rawpath);
    
    TaskQueue::AddTasks([this, cb]{
        // 通知所有网格更新纹理引用
        ModelData::m_MeshesReady.store(true, std::memory_order_release);
        for(auto& mesh : m_meshes) {
            mesh.UpdateTextures(mesh.m_textures); 
        }
        m_IsLoading = false;
        cb();
    }, true);
    std::cout << "---[ModelLoader] 加载模型结束" << std::endl;
}

bool ModelLoader::IsReady() const { return ModelData::m_MeshesReady.load(std::memory_order_acquire); }


} // namespace CubeDemo