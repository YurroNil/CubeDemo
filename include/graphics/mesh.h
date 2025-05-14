// include/graphics/mesh.h

#pragma once
#include <vector>
#include "graphics/shader.h"
#include "resources/texture.h"

namespace CubeDemo {

struct Vertex {
    vec3 Position;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Tangent;
};

// 别名
using TexPtrArray = std::vector<TexturePtr>;
using VertexArray = std::vector<Vertex>;
using UnsignedArray = std::vector<unsigned>;

class Mesh {
public:
    VertexArray Vertices;
    TexPtrArray m_textures;

    // 提供一个默认的构造函数
    Mesh();
    
    // 普通版本的构造函数
    Mesh(const VertexArray& vertices, const UnsignedArray& indices, const TexPtrArray& textures);
        
    // 添加移动构造函数的许可
    Mesh(Mesh&& other) noexcept;

    // 赋值运算符重载为移动操作
    Mesh& operator=(Mesh&& other) noexcept;

    // 禁用拷贝构造函数
    Mesh(const Mesh&) = delete;

    // 禁用拷贝赋值运算符（避免默认浅拷贝）
    Mesh& operator=(const Mesh&) = delete;

    // 构造函数的深拷贝实现(左移运算符重载)
    Mesh& operator<<(const Mesh& other);

    void Draw(Shader& shader) const;
    void UpdateTextures(const TexPtrArray& newTextures);

    // Getters
    const UnsignedArray& GetIndices() const;
    unsigned int GetVAO() const;
    unsigned int GetVBO() const;
    unsigned int GetEBO() const;
    unsigned int GetIndexCount() const;


private:
    unsigned m_VAO, m_VBO, m_EBO;
    size_t m_indexCount;
    mutable std::mutex m_TextureMutex;

    void ReleaseGLResources();
    UnsignedArray m_Indices;
};

}