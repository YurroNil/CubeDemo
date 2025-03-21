// include/rendering/shader.h

#pragma once
#include <string>
#include "core/camera.h"
#include "glm/glm.hpp"

class Shader {
public:
    Shader(const std::string& vertexPath, const std::string& fragmentPath);
    ~Shader();
    
    void Use() const;
    void SetMat4(const std::string& name, const glm::mat4& mat) const;
    void ApplyCamera(const Camera& camera, float aspectRatio) const;
    void SetVec3(const std::string& name, const glm::vec3& value);

private:
    unsigned int ID;
};
