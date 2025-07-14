// src/loaders/model.cpp
#include "pch.h"
#include "loaders/model.h"
#include "loaders/resource.h"
#include "loaders/progress_tracker.h"
#include "resources/model.h"

// 别名
using MaL = CubeDemo::Loaders::Material;
using ML = CubeDemo::Loaders::Model;

namespace CubeDemo {

// 外部变量声明
extern bool DEBUG_ASYNC_MODE; extern unsigned int DEBUG_INFO_LV;

// 静态成员初始化
std::unordered_set<string> ML::m_PrintedPaths;
std::mutex ML::m_PrintMutex; 

ML::Model(const string& path, ::CubeDemo::Model* model)
    : Rawpath(path), m_owner(model)
{
    Directory = path.substr(0, path.find_last_of('/'));
}

void ML::LoadModel(const string& path) {

    // 更新进度：开始加载文件
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_FILE, 
        path, 
        0.1f
    );

/* -------模型加载------- */
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace // 生成切线数据
    );

    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        ProgressTracker::Get().FinishResource(ProgressTracker::MODEL_FILE, path);
        ProgressTracker::Get().FinishResource(ProgressTracker::MODEL_GEOMETRY, path);

        std::cerr << "[Assimp错误] " << importer.GetErrorString() << std::endl; return;
    }

    // 更新文件加载进度
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_FILE, 
        path, 
        0.5f  // 文件已加载
    );
    
    // 更新几何处理开始
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_GEOMETRY, 
        path, 
        0.0f
    );

    Directory = path.substr(0, path.find_last_of('/'));

    // 处理节点时更新进度
    const int totalNodes = CountNodes(scene->mRootNode);
    int processedNodes = 0;
    ProcNode(scene->mRootNode, scene, processedNodes, totalNodes);


    if(DEBUG_INFO_LV > 1) std::cout << "\n=== 模型加载诊断 ===" << "\n    加载模型: " << path << "\n    节点数量: " << scene->mRootNode->mNumChildren << ", 网格数量: " << scene->mNumMeshes << ", 材质数量: " << scene->mNumMaterials << std::endl;

    // 更新几何处理进度
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_GEOMETRY, 
        path, 
        0.8f  // 几何处理中
    );

/* -------计算包围球------- */
    m_owner->bounds.Calc(m_owner->GetMeshes());

    if(DEBUG_INFO_LV > 1) std::cout << ", 总网格数: " << m_owner->GetMeshes().size() << ", 包围球半径: " << m_owner->bounds.Rad << "\n=== 加载完成 ===\n" << std::endl;

    // 完成几何处理
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_GEOMETRY, 
        path, 
        1.0f
    );
    
    // 完成文件加载
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::MODEL_FILE, 
        path, 
        1.0f
    );
}

// 计算节点总数
int ML::CountNodes(aiNode* node) {
    int count = 1; // 当前节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        count += CountNodes(node->mChildren[i]);
    }
    return count;
}

void ML::ProcNode(aiNode* node, const aiScene* scene, int& processedNodes, int totalNodes) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {

        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // 生成原始网格
        Mesh original = ProcMesh(mesh, scene);

        // 移动语义转移所有权
        m_owner->GetMeshes().push_back(std::move(original));
    }

    processedNodes++;
    
    // 每处理10个节点更新一次进度
    if (processedNodes % 10 == 0) {
        float progress = 0.5f + 0.3f * (static_cast<float>(processedNodes) / totalNodes);
        ProgressTracker::Get().UpdateProgress(
            ProgressTracker::MODEL_GEOMETRY, 
            Rawpath, 
            progress
        );
    }

    // 递归处理子节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        ProcNode(node->mChildren[i], scene, processedNodes, totalNodes);
    }
}

Mesh ML::ProcMesh(aiMesh* mesh, const aiScene* scene) {
    VertexArray vertices;
    UnsignedArray indices;
    TexPtrArray textures;
    
    // 处理顶点数据
    for (unsigned i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // 位置
        vertex.Position = vec3(
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
        );
        // 法线
        if (mesh->mNormals)
            vertex.Normal = vec3(
                mesh->mNormals[i].x,
                mesh->mNormals[i].y,
                mesh->mNormals[i].z
            );
        // 纹理坐标（仅处理第一组）
        if (mesh->mTextureCoords[0]) {
            vertex.TexCoords = vec2(
                mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y
            );
        }
        vertices.push_back(vertex);
    }
    // 处理索引数据
    for (unsigned i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }
    
    // 处理材质
     if (mesh->mMaterialIndex >= 0) {

        MaL::ProcMaterial(mesh, scene, textures, DEBUG_ASYNC_MODE);

        // 去重处理
        std::unordered_map<string, TexturePtr> unique_tex;
        for (auto& tex : textures) {
            auto it = unique_tex.find(tex->Path);
            if (it != unique_tex.end()) {
                if(DEBUG_INFO_LV > 1) std::cout << "[模型加载器] 网格内去重纹理: " << tex->Path << std::endl;
            } else {
                unique_tex[tex->Path] = tex;
            }
        }
        textures.clear();
        for (auto& [path, tex] : unique_tex) textures.push_back(tex);
        
    } else std::cerr << "无效材质索引: " << mesh->mMaterialIndex << "/" << scene->mNumMaterials << std::endl;

    // 处理材质
    MaterialPtr material = nullptr;
    if (mesh->mMaterialIndex >= 0) {
        aiMaterial* aiMat = scene->mMaterials[mesh->mMaterialIndex];
        material = MaL::CreateMaterial(aiMat);
    }
    
    return Mesh(vertices, indices, textures, material);
}

// 异步加载模型
void ML::LoadAsync(ModelLoadCallback cb) {
    m_owner->SetMeshMarker() = true;
    RL::EnqueueIOJob([this, cb]{

        if(DEBUG_INFO_LV > 1) std::cout << "\n---[模型加载器] 使用异步加载模式加载模型..." << std::endl;
        this->LoadModel(Rawpath);
        
        TaskQueue::AddTasks([this, cb]{
            // 通知所有网格更新纹理引用
            m_owner->SetMeshMarker().store(true, std::memory_order_release);

            for(auto& mesh : m_owner->GetMeshes()) mesh.UpdateTextures(mesh.m_textures);

            m_owner->SetLoadingMarker() = false; cb();
        }, true);
    });
}

// 同步加载模型
void ML::LoadSync(ModelLoadCallback cb) {
    m_owner->SetLoadingMarker() = true;
    if(DEBUG_INFO_LV > 1) std::cout << "\n---[模型加载器] 使用同步加载模式加载模型..." << std::endl;

    this->LoadModel(Rawpath);
    
    TaskQueue::AddTasks([this, cb]{
        // 通知所有网格更新纹理引用
        m_owner->SetMeshMarker().store(true, std::memory_order_release);
        // for(auto& mesh : m_owner->GetMeshes()) mesh.UpdateTextures(mesh.m_textures);
        m_owner->SetLoadingMarker() = false; cb();
    }, true);

    if(DEBUG_INFO_LV > 1) std::cout << "---[模型加载器] 加载模型结束\n" << std::endl;
}
} // namespace CubeDemo
