#pragma once
#include <string>
#include "core/camera.h"
#include "3rd-lib/glm/glm.hpp"

class Shader {
public:
    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();
    
    void Use() const;
    void SetMat4(const std::string &name, const glm::mat4 &mat) const;
    void ApplyCamera(const Camera& camera, float aspectRatio);

private:
    unsigned int ID;
};
