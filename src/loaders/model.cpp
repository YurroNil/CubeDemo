// src/loaders/model.cpp
#include "pch.h"
#include "loaders/model.h"
#include "loaders/resource.h"

// 别名
using MaL = CubeDemo::Loaders::Material;
using ML = CubeDemo::Loaders::Model;

namespace CubeDemo {

// 外部变量声明
extern bool DEBUG_ASYNC_MODE;
extern bool DEBUG_LOD_MODE;

// 静态成员初始化
std::unordered_set<string> ML::s_PrintedPaths;
std::mutex ML::s_PrintMutex; 

ML::Model(const string& path) : Rawpath(path) {
    Directory = path.substr(0, path.find_last_of('/'));
}

void ML::LoadModel(const string& path) {
/* -------模型加载------- */
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace // 生成切线数据
    );

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[Assimp Error] " << importer.GetErrorString() << std::endl; return;
    }

    Directory = path.substr(0, path.find_last_of('/'));
    ProcNode(scene->mRootNode, scene);

    std::cout << "\n=== 模型加载诊断 ===" << "\n    模型路径: " << path << "\n    节点数量: " << scene->mRootNode->mNumChildren << "\n    网格数量: " << scene->mNumMeshes << "\n    材质数量: " << scene->mNumMaterials << std::endl;


/* -------计算包围球------- */
    bounds.Calc(m_meshes);

    std::cout << "    总网格数: " << m_meshes.size() << "\n    包围球半径: " << bounds.Rad << "\n=== 加载完成 ===\n" << std::endl;

/* -------LOD系统初始化------- */

    if (DEBUG_LOD_MODE == false) return;

    m_LODSystem.Init(
        std::move(m_meshes_copy),  // 转移副本所有权
        bounds.Rad,                // 包围球半径
        {0.3f, 0.6f},              // 简化比例
        [](const Mesh& mesh, float ratio) -> Mesh {
            auto simplified = Graphics::simplify_mesh(mesh, ratio);
            simplified.m_textures = mesh.m_textures;
            
            // 输出简化信息
            std::cout << "[LOD] 生成简化网格: 原顶点数=" << mesh.Vertices.size() << " → 简化后顶点数=" << simplified.Vertices.size() << " | 简化比例=" << 100*ratio << "%\n";
            return simplified;
        }
    );
}

void ML::ProcNode(aiNode* node, const aiScene* scene) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {

        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // 生成原始网格
        Mesh original = ProcMesh(mesh, scene);
        Mesh copy;
        copy << original; // 深拷贝

        // 主列表：移动语义转移所有权
        m_meshes.push_back(std::move(original));

        m_meshes_copy.push_back(std::move(copy));

    }

    // 递归处理子节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        ProcNode(node->mChildren[i], scene);
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
            if (auto it = unique_tex.find(tex->Path); it != unique_tex.end()) {
                std::cout << "[优化] 网格内去重纹理: " << tex->Path << "\n";
            } else {
                unique_tex[tex->Path] = tex;
            }
        }
        textures.clear();
        for (auto& [path, tex] : unique_tex) textures.push_back(tex);
        
    } else {
        std::cerr << "无效材质索引: " << mesh->mMaterialIndex << "/" << scene->mNumMaterials << std::endl;
    }

    return Mesh(vertices, indices, textures);
}

// 异步加载模型
void ML::LoadAsync(ModelLoadCallback cb) {
    m_IsLoading = true;
    RL::EnqueueIOJob([this, cb]{

        std::cout << "\n---[ModelAsyncLoader] 开始加载模型..." << std::endl;
        this->LoadModel(Rawpath);
        
        std::cout << "---[ModelAsyncLoader] 加载模型结束" << std::endl;
        
        TaskQueue::AddTasks([this, cb]{
            // 通知所有网格更新纹理引用
            ML::m_MeshesReady.store(true, std::memory_order_release);

            for(auto& mesh : m_meshes) mesh.UpdateTextures(mesh.m_textures);

            m_IsLoading = false; cb();
        }, true);
    });
}

// 同步加载模型
void ML::LoadSync(ModelLoadCallback cb) {
    m_IsLoading = true;
    std::cout << "\n---[模型加载器] 使用同步加载模式加载模型..." << std::endl;

    this->LoadModel(Rawpath);
    
    TaskQueue::AddTasks([this, cb]{
        // 通知所有网格更新纹理引用
        ML::m_MeshesReady.store(true, std::memory_order_release);
        // for(auto& mesh : m_meshes) mesh.UpdateTextures(mesh.m_textures);
        m_IsLoading = false; cb();
    }, true);

    if (DEBUG_LOD_MODE == false) {
        std::cout << "---[模型加载器] 加载模型结束\n" << std::endl;
        return;
    }

    std::cout << "\n---[模型加载器] 生成层次结构...\n" << std::endl;

    const std::vector<float> lod_ratios{0.3f, 0.6f};
    m_LODSystem.GenLodHierarchy_sync(
        m_LODSystem.GetLevelByIndex(0),
        lod_ratios
    );

    std::cout << "---[模型加载器] 加载模型结束\n" << std::endl;
}

// 乱七八糟的Getters
bool ML::IsReady() const { return ML::m_MeshesReady.load(std::memory_order_acquire); }
const Graphics::LODSystem* ML::GetLODSystem() const { return &m_LODSystem; }
const std::atomic<bool>& ML::isLoading() const { return m_IsLoading; }
const MeshArray& ML::GetMeshes() const { return m_meshes; }
const mat4& ML::GetModelMatrix() const { return m_ModelMatrix; }


} // namespace CubeDemo
