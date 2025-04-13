// include/resources/model.h

#pragma once
#include <vector>
#include "graphics/mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

namespace CubeDemo {


class Model {
public:
    Model(const std::string& path);
    void Draw(Shader& shader);
    
private:
    std::vector<Mesh> m_meshes;
    std::string m_directory;
    
    void LoadModel(const std::string& path);
    void ProcessNode(aiNode* node, const aiScene* scene);
    Mesh ProcessMesh(aiMesh* mesh, const aiScene* scene);
    std::vector<std::shared_ptr<Texture>> LoadMaterialTextures(
        aiMaterial* mat, 
        aiTextureType type,
        const std::string& typeName
    );
    
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_textureCache;
};

}