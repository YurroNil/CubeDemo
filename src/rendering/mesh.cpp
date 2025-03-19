#include <cstddef>

#include "rendering/mesh.h"
#include "glad/glad.h"
#include "rendering/modelLoader.h"

Mesh::Mesh(const ModelData& modelData) 
    : _vertices(modelData.vertices),
      _vertexComponents(modelData.vertexComponents) 
{
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

    glEnableVertexAttribArray(0);
}


//图形绘制
void Mesh::Draw() const {
    glBindVertexArray(VAO);
    // 动态计算顶点数量
    int vertexCount = _vertices.size() / _vertexComponents;
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}

Mesh::~Mesh() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}