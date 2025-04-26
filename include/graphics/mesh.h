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
    TexPtrArray m_textures;

    Mesh(const VertexArray& vertices, 
         const std::vector<unsigned>& indices,
         const TexPtrArray& textures);
        
    // 添加移动构造函数
    Mesh(Mesh&& other) noexcept 
        : Vertices(std::move(other.Vertices)),
          m_textures(std::move(other.m_textures)),
          VAO(other.VAO),
          VBO(other.VBO),
          EBO(other.EBO),
          indexCount(other.indexCount)
    {
        // 将源对象置为无效状态
        other.VAO = other.VBO = other.EBO = 0;
        other.indexCount = 0;
    }

    // 添加移动赋值运算符
    Mesh& operator=(Mesh&& other) noexcept {
        if(this != &other) {
            ReleaseGLResources();
            
            Vertices = std::move(other.Vertices);
            m_textures = std::move(other.m_textures);
            VAO = other.VAO;
            VBO = other.VBO;
            EBO = other.EBO;
            indexCount = other.indexCount;

            other.VAO = other.VBO = other.EBO = 0;
            other.indexCount = 0;
        }
        return *this;
    }

    // 删除拷贝构造和拷贝赋值
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

     
    void Draw(Shader& shader) const;

    void UpdateTextures(const TexPtrArray& newTextures);

private:
    unsigned VAO, VBO, EBO;
    size_t indexCount;
    mutable std::mutex m_TextureMutex;
    void ReleaseGLResources();
};

}