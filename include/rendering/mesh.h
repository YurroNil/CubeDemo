#pragma once
#include <vector>
#include "core/modelLoader.h" // 必须包含ModelData定义

class Mesh {
public:
    Mesh(const float* vertices, size_t size);
    //添加explicit防止构造函数被用作类型隐式转换
    explicit Mesh(const ModelData& modelData);
    ~Mesh();
    
    void Draw() const;

private:
    unsigned int VAO, VBO;
    std::vector<float> _vertices;
    int _vertexComponents;
};
