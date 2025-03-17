#pragma once
#include <cstddef>

class Mesh {
public:
    Mesh(const float* vertices, size_t size);
    ~Mesh();
    void Draw() const;

private:
    unsigned int VAO, VBO;
};
