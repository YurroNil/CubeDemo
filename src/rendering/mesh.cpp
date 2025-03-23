// src/rendering/mesh.cpp

#include <cstddef>
#include "rendering/mesh.h"
#include "glad/glad.h"
#include "rendering/modelLoader.h"

void Mesh::BindArrayBuffer(const ModelData& modelData) {
    // 绑定 VAO/VBO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        modelData.vertices.size() * sizeof(float),
        modelData.vertices.data(),
        GL_STATIC_DRAW
    );

    // 根据vertexComponents设置属性
    glVertexAttribPointer(
        0, 
        _vertexComponents,
        GL_FLOAT, 
        GL_FALSE,
        _vertexComponents * sizeof(float),
        (void*)0
    );
}

Mesh::Mesh(const ModelData& modelData) 
    : _vertices(modelData.vertices),
      _vertexComponents(modelData.vertexComponents) 
{
    
    BindArrayBuffer(modelData);
    glEnableVertexAttribArray(0);


    // 添加法线缓冲
    glGenBuffers(1, &NBO);
    glBindBuffer(GL_ARRAY_BUFFER, NBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        modelData.normals.size()*sizeof(float), 
        modelData.normals.data(),
        GL_STATIC_DRAW
    );

    // 设置属性指针
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
}


//图形绘制
void Mesh::Draw() const {
    glBindVertexArray(VAO);
    // 动态计算顶点数量
    int vertexCount = _vertices.size() / _vertexComponents;
    glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount);
}

Mesh::~Mesh() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}