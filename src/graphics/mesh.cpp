// src/graphics/mesh.cpp

#include "glad/glad.h"
#include "kits/strings.h"
#include "graphics/mesh.h"
#include <iostream>

namespace CubeDemo {
extern bool DEBUG_ASYNC_MODE;

Mesh::Mesh(const VertexArray& vertices, const UnsignedArray& indices, const TexPtrArray& textures) 
    : indexCount(indices.size()) {
    
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

    std::cout << "\n=== 创建OpenGL资源 ===" << "\nVAO: " << VAO << "\nVBO: " << VBO << "\nEBO: " << EBO << "\n顶点数: " << vertices.size() << "\n索引数: " << indices.size() << "\n纹理数: " << textures.size() << std::endl;

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
    // std::cout << "\n[绘制调用] VAO: " << VAO << " | 索引数: " << indexCount << " | 纹理数: " << m_textures.size() << std::endl;

    // 检查VAO有效性
    if(VAO == 0) {
        std::cerr << "[错误] 无效的VAO!" << std::endl;
        return;
    }

    unsigned int diffuseNr = 1, specularNr = 1;
    
    for (size_t i = 0; i < m_textures.size(); ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        
        const auto& tex = m_textures[i];
        string name = tex->Type;
        string number;
        
        if (name == "texture_diffuse") {
            number = std::to_string(diffuseNr++);
        } else if (name == "texture_specular") {
            number = std::to_string(specularNr++);
        }
        
        shader.SetInt((name + number).c_str(), i);
        tex->Bind(i);
    }
    
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    glActiveTexture(GL_TEXTURE0);
    // 绘制后检查错误
    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "[OpenGL错误] 绘制后: 0x" << std::hex << err << std::dec << std::endl;
    }

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