// include/graphics/mesh.h

#pragma once
#include "resources/modelLoader.h"

class Mesh {
public:
    void BindArrayBuffer(const ModelData& modelData);
    Mesh(const float* vertices, size_t size);
    //添加explicit防止构造函数被用作类型隐式转换
    explicit Mesh(const ModelData& modelData);
    
    ~Mesh();
    
    void Draw() const;

private:
    unsigned int VAO, VBO, NBO;
    std::vector<float> _vertices;
    int _vertexComponents;
};
