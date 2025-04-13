// src/resources/model.cpp
#include "resources/model.h"
#include "utils/streams.h"
#include <filesystem>
namespace fs = std::filesystem;


namespace CubeDemo {
Model::Model(const string& path) {
    LoadModel(path);
}

void Model::LoadModel(const string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, 
        aiProcess_Triangulate | 
        aiProcess_FlipUVs | 
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace // 生成切线数据
    );
    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[Assimp Error] " << importer.GetErrorString() << std::endl;
        return;
    }
    
    m_directory = path.substr(0, path.find_last_of('/'));
    ProcessNode(scene->mRootNode, scene);


}

void Model::ProcessNode(aiNode* node, const aiScene* scene) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(ProcessMesh(mesh, scene));
    }
    
    // 递归处理子节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        ProcessNode(node->mChildren[i], scene);
    }
}

Mesh Model::ProcessMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned> indices;
    std::vector<std::shared_ptr<Texture>> textures;
    
    // 处理顶点数据
    for (unsigned i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // 位置
        vertex.Position = glm::vec3(
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
        );
        // 法线
        if (mesh->mNormals)
            vertex.Normal = glm::vec3(
                mesh->mNormals[i].x,
                mesh->mNormals[i].y,
                mesh->mNormals[i].z
            );
        // 纹理坐标（仅处理第一组）
        if (mesh->mTextureCoords[0]) {
            vertex.TexCoords = glm::vec2(
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
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];


        // 漫反射贴图
        auto diffuseMaps = LoadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
        if (diffuseMaps.empty()) { // 如果未找到，尝试其他类型
            diffuseMaps = LoadMaterialTextures(material, aiTextureType_BASE_COLOR, "texture_diffuse");
        }
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());


         // 加载反射贴图
        auto reflectionMaps = LoadMaterialTextures(material, aiTextureType_REFLECTION, "texture_reflection");
        textures.insert(textures.end(), reflectionMaps.begin(), reflectionMaps.end());


        // 法线贴图（map_Bump）
        auto normalMaps = LoadMaterialTextures(material, aiTextureType_NORMALS, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());


        // 高光贴图（map_Ns）
        auto specularMaps = LoadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
        if (specularMaps.empty()) { // 如果未找到，尝试其他类型
            specularMaps = LoadMaterialTextures(material, aiTextureType_SHININESS, "texture_diffuse");
        }
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());


        // 环境光遮蔽（map_Ka）
        auto aoMaps = LoadMaterialTextures(material, aiTextureType_AMBIENT, "texture_ao");
        textures.insert(textures.end(), aoMaps.begin(), aoMaps.end());
    }

    return Mesh(vertices, indices, textures);
}


std::vector<std::shared_ptr<Texture>> Model::LoadMaterialTextures(
    aiMaterial* mat, 
    aiTextureType type,
    const string& typeName) 
{

    std::vector<std::shared_ptr<Texture>> textures;
    const unsigned textureCount = mat->GetTextureCount(type);

    string tempPath;
    for (unsigned i = 0; i < textureCount; ++i) {
        aiString str;
        mat->GetTexture(type, i, &str);
        
        // 处理Blender的特殊参数
        std::string path(str.C_Str());

        const string fullPath = m_directory + "/textures/" + fs::path(path).filename().string();
        tempPath = fullPath;
        std::cout << "[DEBUG] 正在处理纹理: " << fullPath << "(" << typeName << ", 数量: " << textureCount << "), aiTextureTyp枚举值: " << type << std::endl;
        
        // 使用纹理工厂方法
        try {
            auto tex = Texture::Create(fullPath, typeName);
            textures.push_back(tex);
            
            // 缓存到模型层级
            if (!m_textureCache.count(tex->Path)) {
                m_textureCache[tex->Path] = tex;
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] 材质加载失败: " << e.what() << std::endl;
            return {};  // 返回空集合
        }
        std::cout << "[DEBUG] 此纹理已被成功加载. " << std::endl;

    }

    return textures;
}




void Model::Draw(Shader& shader) {
    for (auto& mesh : m_meshes) {
        mesh.Draw(shader);
    }
}

}