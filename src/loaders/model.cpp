// src/loaders/model.cpp

#include "loaders/model.h"
// 标准库
#include "kits/streams.h"

using MaL = CubeDemo::Loaders::Material;
using ML = CubeDemo::Loaders::Model;
namespace CubeDemo {
extern bool DEBUG_ASYNC_MODE;


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

     std::cout << "\n=== 模型加载诊断 ===" << "\n模型路径: " << path << "\n节点数量: " << scene->mRootNode->mNumChildren << "\n网格数量: " << scene->mNumMeshes << "\n材质数量: " << scene->mNumMaterials << std::endl;

    ProcNode(scene->mRootNode, scene);
     std::cout << "=== 加载完成 ===" << "\n总网格数: " << m_meshes.size() << "\n包围球半径: " << bounds.Rad << std::endl;

/* -------计算包围球------- */
    bounds.Calc(m_meshes);

}

void ML::ProcNode(aiNode* node, const aiScene* scene) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(std::move(ProcMesh(mesh, scene)));
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
        std::unordered_map<string, TexturePtr> uniqueTextures;
        for (auto& tex : textures) {
            if (auto it = uniqueTextures.find(tex->Path); it != uniqueTextures.end()) {
                std::cout << "[优化] 网格内去重纹理: " << tex->Path << "\n";
            } else {
                uniqueTextures[tex->Path] = tex;
            }
        }
        textures.clear();
        for (auto& [path, tex] : uniqueTextures) textures.push_back(tex);
        

        std::cout << "[材质处理] 完成，加载纹理数: " << textures.size() << std::endl;
    } else {
        std::cerr << "无效材质索引: " << mesh->mMaterialIndex << "/" << scene->mNumMaterials << std::endl;
    }
    
    return Mesh(vertices, indices, textures);
}

// 渲染循环中绘制模型
void ML::Draw(Shader& shader) {
    shader.SetMat4("model", m_ModelMatrix);

    if(m_IsLoading.load()) return; // 加载中不绘制
    // 绘制模型的所有网格
    for (auto& mesh : m_meshes) { mesh.Draw(shader); }
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

// 同步加载模型（调试专用）
void ML::LoadSync(ModelLoadCallback cb) {
    m_IsLoading = true;
    std::cout << "\n---[模型加载器] 使用同步加载模式加载模型..." << std::endl;

    this->LoadModel(Rawpath);
    
    TaskQueue::AddTasks([this, cb]{
        // 通知所有网格更新纹理引用
        ML::m_MeshesReady.store(true, std::memory_order_release);
        for(auto& mesh : m_meshes) mesh.UpdateTextures(mesh.m_textures);
        
        m_IsLoading = false;
        cb();
    }, true);
    std::cout << "---[模型加载器] 加载模型结束" << std::endl;
}

bool ML::IsReady() const { return ML::m_MeshesReady.load(std::memory_order_acquire); }

} // namespace CubeDemo