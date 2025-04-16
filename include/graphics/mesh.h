// include/graphics/mesh.h

#pragma once
#include <vector>
#include "graphics/shader.h"
#include "resources/texture.h"

namespace CubeDemo {
    // 乱七八糟的前置声明
struct Vertex; using TexPtrArray = std::vector< TexturePtr >; using VertexArray = std::vector<Vertex>;


struct Vertex { // 声明Vertex结构体
    vec3 Position;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Tangent;
};

class Mesh {    // Mesh类
public:
    VertexArray Vertices;

    Mesh(const VertexArray& vertices, 
         const std::vector<unsigned>& indices,
         const TexPtrArray& textures);
     
    void Draw(Shader& shader) const;

private:
    unsigned VAO, VBO, EBO;
    size_t indexCount;
    TexPtrArray m_textures;
};

}