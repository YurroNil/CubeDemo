// include/graphics/mesh.h

#pragma once
#include <vector>
#include "graphics/shader.h"
#include "resources/texture.h"


namespace CubeDemo {

struct Vertex {
    vec3 Position;
    vec3 Normal;
    glm::vec2 TexCoords;
    vec3 Tangent;
};


class Mesh {
public:
    std::vector<Texture> textures;
    

    Mesh(const std::vector<Vertex>& vertices, 
         const std::vector<unsigned>& indices,
         const std::vector<std::shared_ptr<Texture>>& textures);
     
    void Draw(Shader& shader) const;

private:
    unsigned VAO, VBO, EBO;
    size_t indexCount;
    std::vector<std::shared_ptr<Texture>> m_textures;
};

}