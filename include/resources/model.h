// include/resources/model.h

#pragma once
#include <vector>
#include "graphics/mesh.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

namespace CubeDemo {
// 乱七八糟的前置声明
using TexPtrArray = std::vector< TexturePtr >; using VertexArray = std::vector<Vertex>; using MeshArray = std::vector<Mesh>;


// 包围球
struct BoundingSphere {
    vec3 Center;    // 包围球中心
    float Rad;      // 半径
    // 计算包围球
    void Calc(const MeshArray& meshes);
};


// Model类
class Model {
public:
    Model(const string& path);
    void Draw(Shader& shader);

    // 包维球实例
    BoundingSphere bounds;
    MeshArray m_meshes;
    
private:

    string m_directory;
    
    void LoadModel(const string& path);
    void ProcNode(aiNode* node, const aiScene* scene);
    Mesh ProcMesh(aiMesh* mesh, const aiScene* scene);
    TexPtrArray LoadMaterialTex(aiMaterial* mat, aiTextureType type, const string& typeName );

};

}   // namespace CubeDemo