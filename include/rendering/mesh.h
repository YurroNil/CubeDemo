#pragma once
#include <vector>
#include "rendering/modelLoader.h"

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
