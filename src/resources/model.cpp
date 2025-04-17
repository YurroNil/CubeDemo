// src/resources/model.cpp

// 标准库
#include "utils/streams.h"
#include "utils/fileSystemKits.h"
// 第三方库
#include "stb_image.h"
// 项目头文件
#include "resources/model.h"
#include "core/window.h"


namespace CubeDemo {

Model::Model(const string& path) {
    LoadModel(path);
}

void Model::LoadModel(const string& path) {
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
    m_directory = path.substr(0, path.find_last_of('/'));
    ProcNode(scene->mRootNode, scene);

/* -------计算包围球------- */
    bounds.Calc(m_meshes);

}

void Model::ProcNode(aiNode* node, const aiScene* scene) {
    // 处理当前节点的所有网格
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(ProcMesh(mesh, scene));
    }
    
    // 递归处理子节点
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        ProcNode(node->mChildren[i], scene);
    }
}

Mesh Model::ProcMesh(aiMesh* mesh, const aiScene* scene) {
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
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];


        // 漫反射贴图
        auto diffuseMaps = LoadMaterialTex(material, aiTextureType_DIFFUSE, "texture_diffuse");
        if (diffuseMaps.empty()) { // 如果未找到，尝试其他类型
            diffuseMaps = LoadMaterialTex(material, aiTextureType_BASE_COLOR, "texture_diffuse");
        }
        textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());


         // 加载反射贴图
        auto reflectionMaps = LoadMaterialTex(material, aiTextureType_REFLECTION, "texture_reflection");
        textures.insert(textures.end(), reflectionMaps.begin(), reflectionMaps.end());


        // 法线贴图（map_Bump）
        auto normalMaps = LoadMaterialTex(material, aiTextureType_NORMALS, "texture_normal");
        textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());


        // 高光贴图（map_Ns）
        auto specularMaps = LoadMaterialTex(material, aiTextureType_SPECULAR, "texture_specular");
        if (specularMaps.empty()) { // 如果未找到，尝试其他类型
            specularMaps = LoadMaterialTex(material, aiTextureType_SHININESS, "texture_diffuse");
        }
        textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

        // 环境光遮蔽（map_Ka）
        auto aoMaps = LoadMaterialTex(material, aiTextureType_AMBIENT, "texture_ao");
        textures.insert(textures.end(), aoMaps.begin(), aoMaps.end());
    }

    return Mesh(vertices, indices, textures);
}

TexPtrArray Model::LoadMaterialTex(aiMaterial* mat, aiTextureType type, const string& typeName) {
    TexPtrArray textures;
    const unsigned textureCount = mat->GetTextureCount(type);

    string tempPath;
    for (unsigned i = 0; i < textureCount; i++) {

        aiString str; mat->GetTexture(type, i, &str); string path(str.C_Str());

        const string fullPath = m_directory + "/textures/" + fs::path(path).filename().string();
        tempPath = fullPath;
        std::cout << "--------------------\n" << "[DEBUG] 正在处理纹理: " << fullPath << "(" << typeName << ", 数量: " << textureCount << "), aiTextureTyp枚举值: " << type << std::endl;
        
        // 使用纹理工厂方法
        try {
            // 修改为异步加载
            auto tex = Texture::Create(fullPath, typeName);

            if(!tex || tex->ID == 0) { throw std::runtime_error("[ERROR] 纹理创建返回空对象"); }
            textures.push_back(tex);
            // 实时引用计数检查
        std::cout << "  当前纹理引用计数: " << tex.use_count() << ", 类型: " << typeName << ". " << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] 材质加载失败: " << e.what();
            std::cerr << ", 当前处理文件: " << fullPath << ", 关联模型: " << m_directory << std::endl;
            throw;
        }
        std::cout << "[DEBUG] 此纹理已被成功加载. " << std::endl;
    }
    return textures; // 返回的textures将持有纹理引用
}

void Model::Draw(Shader& shader) {
    // 绘制模型的所有网格
    for (auto& mesh : m_meshes) { mesh.Draw(shader); }
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

}