// src/graphics/mesh.cpp

#include "glad/glad.h"
#include "graphics/mesh.h"
#include <iostream>

namespace CubeDemo {
extern bool DEBUG_ASYNC_MODE;

// 普通版本的构造函数
Mesh::Mesh(
    const VertexArray& vertices,
    const UnsignedArray& indices,
    const TexPtrArray& textures
)
    : m_textures(textures),
      m_indexCount(indices.size()),
      m_Indices(indices)

{

    // 储存顶点数组
    this->Vertices = vertices;

    // 各种VAO、VBO和EBO的绑定
    glGenVertexArrays(1, &m_VAO);
    
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), &indices[0], GL_STATIC_DRAW);

    // 顶点属性
    // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    
    // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    
    // TexCoords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    
    // Tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));

    glBindVertexArray(0);

    // 添加OpenGL错误检查
    GLenum err; static bool notice = false;

    while(notice == false && ((err = glGetError()) != GL_NO_ERROR))
    {
        notice = true;
        std::cerr << "[OpenGL错误] 在Mesh构造函数中: 0x" << std::hex << err << std::dec << std::endl;
    }
}

// 提供一个默认的构造函数
Mesh::Mesh() 
    : m_textures{},    // 设置为空容器
      m_Indices{},     // 设置为空容器
      m_indexCount(0), m_VAO(0), m_VBO(0), m_EBO(0) // 设置为空数据
{}

// 纹理更新
void Mesh::UpdateTextures(const TexPtrArray& newTextures) {
    if(DEBUG_ASYNC_MODE) std::lock_guard lock(m_TextureMutex);
    m_textures = newTextures;
}

// 渲染循环中网格绘制
void Mesh::Draw(Shader& shader) const {

     // 检查VAO有效性
    if(m_VAO == 0) {
        std::cerr << "[错误] 无效的VAO!" << std::endl;
        return;
    }
     // 在绑定纹理前启用混合
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 标准混合模式

    // 计数器初始化
    unsigned int
        diffuse_count = 1,
        specular_count = 1,
        normal_count = 1,
        ao_count = 1;


     // 遍历所有纹理
    for (size_t i = 0; i < m_textures.size(); ++i) {

        if(!m_textures[i] || !m_textures[i]->m_Valid.load()) {
            std::cerr << "[警告] 跳过无效纹理: 索引" << i << std::endl;
            continue;
        }

        glActiveTexture(GL_TEXTURE0 + i); // 激活对应纹理单元

        const auto& tex = m_textures[i];
        string type = tex->Type;
        string uniform_name;

        // 动态生成uniform名称
        if (type == "texture_diffuse") {
            uniform_name = type + std::to_string(diffuse_count++);
        } else if (type == "texture_specular") {
            uniform_name = type + std::to_string(specular_count++);
        } else if (type == "texture_normal") {
            uniform_name = type + std::to_string(normal_count++);
        } else if (type == "texture_ao") {
            uniform_name = type + std::to_string(ao_count++);
        } else {
            uniform_name = type + "_unknown";
        }

        // 设置Shader参数并绑定纹理
        shader.SetInt(uniform_name.c_str(), i);
        tex->Bind(i);
    }

    // 绘制网格
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // 重置纹理单元
    glActiveTexture(GL_TEXTURE0);

}

// 构造函数的移动实现
Mesh::Mesh(Mesh&& other) noexcept
    : Vertices(std::move(other.Vertices)),
      m_Indices(std::move(other.m_Indices)),
      m_textures(std::move(other.m_textures)),
      m_VAO(other.m_VAO), m_VBO(other.m_VBO), m_EBO(other.m_EBO),
      m_indexCount(other.m_indexCount) 
{
    other.m_VAO = other.m_VBO = other.m_EBO = 0;
    other.m_indexCount = 0;
}

// 将等号重载为移动
Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if(this != &other) {
        ReleaseGLResources();
        
        Vertices = std::move(other.Vertices);
        m_textures = std::move(other.m_textures);
        m_VAO = other.m_VAO;
        m_VBO = other.m_VBO;
        m_EBO = other.m_EBO;
        m_indexCount = other.m_indexCount;

        other.m_VAO = other.m_VBO = other.m_EBO = 0;
        other.m_indexCount = 0;
    }
    return *this;
}


// 构造函数的深拷贝实现(左移运算符重载)
Mesh& Mesh::operator<<(const Mesh& other) {
    // 将other的数据拷贝到this这里
    Vertices = other.Vertices;
    m_Indices = other.m_Indices;
    m_textures = other.m_textures;
    m_indexCount = other.m_indexCount;

    // 重新生成OpenGL资源
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);

    // 绑定并复制顶点数据
    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, Vertices.size() * sizeof(Vertex), Vertices.data(), GL_STATIC_DRAW);

    // 绑定并复制索引数据
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_Indices.size() * sizeof(unsigned), m_Indices.data(), GL_STATIC_DRAW);

    // 设置顶点属性指针（与构造函数一致）
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));

    glBindVertexArray(0);

    // 检查OpenGL错误
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[OpenGL Error] 在拷贝构造函数中: 0x" << std::hex << err << std::dec << std::endl;
    }
    return *this;
}


void Mesh::ReleaseGLResources() {
    if(m_VAO) glDeleteVertexArrays(1, &m_VAO);
    if(m_VBO) glDeleteBuffers(1, &m_VBO);
    if(m_EBO) glDeleteBuffers(1, &m_EBO);
}



// 乱七八糟的Getters
const UnsignedArray& Mesh::GetIndices() const { return m_Indices; }
unsigned int Mesh::GetVAO() const { return m_VAO; }
unsigned int Mesh::GetVBO() const { return m_VBO; }
unsigned int Mesh::GetEBO() const { return m_EBO; }
unsigned int Mesh::GetIndexCount() const { return m_indexCount; }

}