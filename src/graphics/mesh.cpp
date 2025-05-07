// src/graphics/mesh.cpp

#include "glad/glad.h"
#include "kits/strings.h"
#include "graphics/mesh.h"
#include <iostream>

namespace CubeDemo {
extern bool DEBUG_ASYNC_MODE;

Mesh::Mesh(const VertexArray& vertices, const UnsignedArray& indices, const TexPtrArray& textures) 
    : m_textures(textures), indexCount(indices.size()) {
    
    // 储存顶点数组
    this->Vertices = vertices;

    // 各种VAO、VBO和EBO的绑定
    glGenVertexArrays(1, &VAO);
    
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
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
    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[OpenGL错误] 在Mesh构造函数中: 0x" << std::hex << err << std::dec << std::endl;
    }

}

void Mesh::UpdateTextures(const TexPtrArray& newTextures) {
    if(DEBUG_ASYNC_MODE) std::lock_guard lock(m_TextureMutex);
    m_textures = newTextures;
}

void Mesh::Draw(Shader& shader) const {

     // 检查VAO有效性
    if(VAO == 0) {
        std::cerr << "[错误] 无效的VAO!" << std::endl;
        return;
    }

    // 计数器初始化
    unsigned int diffuseCount = 1;
    unsigned int specularCount = 1;
    unsigned int normalCount = 1;
    unsigned int aoCount = 1;

    static int counter = 0;
     // 遍历所有纹理
    if(counter==0) {
        std::cout << "[Mesh] m_textures数量: " << m_textures.size() << std::endl;
        counter++;
    }

    for (size_t i = 0; i < m_textures.size(); ++i) {

        if(!m_textures[i] || !m_textures[i]->m_Valid.load()) {
            std::cerr << "[警告] 跳过无效纹理: 索引" << i << std::endl;
            continue;
        }

        glActiveTexture(GL_TEXTURE0 + i); // 激活对应纹理单元

        const auto& tex = m_textures[i];
        std::string type = tex->Type;
        std::string uniformName;

        // 动态生成uniform名称
        if (type == "texture_diffuse") {
            uniformName = type + std::to_string(diffuseCount++);
        } else if (type == "texture_specular") {
            uniformName = type + std::to_string(specularCount++);
        } else if (type == "texture_normal") {
            uniformName = type + std::to_string(normalCount++);
        } else if (type == "texture_ao") {
            uniformName = type + std::to_string(aoCount++);
        } else {
            uniformName = type + "_unknown";
        }

        // 设置Shader参数并绑定纹理
        shader.SetInt(uniformName.c_str(), i);
        tex->Bind(i);
    }

    // 绘制网格
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // 重置纹理单元
    glActiveTexture(GL_TEXTURE0);

}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
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

void Mesh::ReleaseGLResources() {
    if(VAO) glDeleteVertexArrays(1, &VAO);
    if(VBO) glDeleteBuffers(1, &VBO);
    if(EBO) glDeleteBuffers(1, &EBO);
}


}