// src/resources/modelData.cpp

// 第三方库
#include "stb_image.h"
// 项目头文件
#include "threads/modelLoader.h"

namespace CubeDemo {

// ModelData类方法实现
ModelData::ModelData(const string& path) : MaterialData(), Rawpath(path) {
    Directory = path.substr(0, path.find_last_of('/'));
}

void ModelData::LoadModel(const string& path) {
/* -------模型加载------- */
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace // 生成切线数据
    );
    std::cout << "[断点A]" << std::endl;

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[Assimp Error] " << importer.GetErrorString() << std::endl; return;
    }

    Directory = path.substr(0, path.find_last_of('/'));
    ProcNode(scene->mRootNode, scene);
    std::cout << "[断点ZZZ]" << std::endl;
/* -------计算包围球------- */
    bounds.Calc(m_meshes);

}

void ModelData::ProcNode(aiNode* node, const aiScene* scene) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(std::move(ProcMesh(mesh, scene)));
    }
    std::cout << "[断点B]" << std::endl;
    // 递归处理子节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        ProcNode(node->mChildren[i], scene);
    }
}

Mesh ModelData::ProcMesh(aiMesh* mesh, const aiScene* scene) {
    VertexArray vertices;
    std::vector<unsigned> indices;
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
        MaterialData::ProcMaterial(mesh, scene, textures);
    }

    return Mesh(vertices, indices, textures);
}

// 渲染循环中绘制模型
void ModelData::Draw(Shader& shader) {
    if(m_IsLoading.load()) return; // 加载中不绘制
    // 绘制模型的所有网格
    for (auto& mesh : m_meshes) {
        if(mesh.m_textures.empty()) return; // 无纹理不绘制
        mesh.Draw(shader);
        }
}

// 计算包围球
void BoundingSphere::Calc(const MeshArray& meshes) {
    if (meshes.empty()) {
        Center = vec3(0.0f);
        Rad = 0.0f;
        return;
    }
    // 计算AABB
    vec3 minVert(FLT_MAX), maxVert(-FLT_MAX);
    for (const auto& mesh : meshes) {
        for (const auto& vert : mesh.Vertices) {
            minVert = glm::min(minVert, vert.Position);
            maxVert = glm::max(maxVert, vert.Position);
        }
    }
    // 中心点计算
    Center = (minVert + maxVert) * 0.5f;
    
    // 计算最大半径
    float maxDist = 0.0f;
    for (const auto& mesh : meshes) {
        for (const auto& vert : mesh.Vertices) {
            maxDist = glm::max(maxDist, glm::length(vert.Position - Center));
        }
    }
    Rad = maxDist;
}

}   // namespace CubeDemo