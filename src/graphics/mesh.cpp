// src/graphics/mesh.cpp

#include <cstddef>
#include "graphics/mesh.h"
#include "glad/glad.h"
#include "resources/modelLoader.h"

// 绑定顶点缓冲对象并配置顶点属性
void Mesh::BindArrayBuffer(const ModelData& modelData) {
    // 生成并绑定顶点数组对象(VAO)
    glGenVertexArrays(1, &VAO);      // 创建1个VAO，ID存入VAO变量
    glGenBuffers(1, &VBO);           // 创建1个顶点缓冲对象(VBO)
    glBindVertexArray(VAO);          // 绑定VAO，后续操作关联到此VAO
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // 绑定VBO到GL_ARRAY_BUFFER目标

    // 将顶点数据上传到GPU
    glBufferData(
        GL_ARRAY_BUFFER,             // 目标缓冲类型
        modelData.vertices.size() * sizeof(float), // 数据大小
        modelData.vertices.data(),   // 数据指针
        GL_STATIC_DRAW               // 使用提示：数据不会频繁修改
    );

    // 配置顶点属性指针（位置属性）
    glVertexAttribPointer(
        0,                           // 属性位置0（对应着色器layout(location=0)）
        _vertexComponents,           // 每个顶点的分量数（如3=xyz）
        GL_FLOAT,                    // 数据类型
        GL_FALSE,                    // 是否标准化
        _vertexComponents * sizeof(float), // 步长（连续顶点间的字节数）
        (void*)(3 * sizeof(float))   // 偏移量（这里可能需要确认数据布局）
        // 注意：偏移量3*sizeof(float)可能表示跳过法线数据？
        // 需确保与ModelData的实际布局一致
    );
}

// 网格构造函数
Mesh::Mesh(const ModelData& modelData) 
    : _vertices(modelData.vertices),
      _vertexComponents(modelData.vertexComponents) 
{
    BindArrayBuffer(modelData);      // 初始化顶点缓冲
    glEnableVertexAttribArray(0);    // 启用属性位置0

    // 添加法线缓冲对象(NBO)
    glGenBuffers(1, &NBO);           // 生成法线缓冲对象
    glBindBuffer(GL_ARRAY_BUFFER, NBO); // 绑定到GL_ARRAY_BUFFER
    glBufferData(
        GL_ARRAY_BUFFER,
        modelData.normals.size()*sizeof(float), // 法线数据量
        modelData.normals.data(),    // 法线数据指针
        GL_STATIC_DRAW               // 数据使用提示
    );

    // 配置法线属性指针（属性位置1）
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);    // 启用属性位置1
}

// 图形绘制方法
void Mesh::Draw() const {
    glBindVertexArray(VAO);          // 绑定预配置的VAO
    // 动态计算顶点数量（总数据量 / 每个顶点分量数）
    int vertexCount = _vertices.size() / _vertexComponents;
    glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount); // 绘制三角形条带
}

// 析构函数：清理OpenGL资源
Mesh::~Mesh() {
    glDeleteVertexArrays(1, &VAO);   // 删除VAO
    glDeleteBuffers(1, &VBO);        // 删除VBO
    // 注意：这里没有删除NBO！会导致内存泄漏
    // 建议添加：glDeleteBuffers(1, &NBO);
}