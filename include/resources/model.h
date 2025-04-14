// include/resources/model.h

#pragma once
#include <vector>
#include "graphics/mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

namespace CubeDemo {
// 乱七八糟的前置声明
using TexturePtrArray = std::vector< std::shared_ptr<Texture> >; using VertexArray = std::vector<Vertex>; 

// Model类
class Model {
public:
    Model(const string& path);
    void Draw(Shader& shader);
    
private:
    std::vector<Mesh> m_meshes;
    string m_directory;
    
    void LoadModel(const string& path);
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);
    TexturePtrArray LoadMaterialTextures(
        aiMaterial* mat, 
        aiTextureType type,
        const string& typeName
    );
    
    TexturePtrHashMap m_textureCache;
};

}