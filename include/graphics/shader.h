// include/renderer/main

#pragma once
#include "core/camera.h"
#include "graphics/shaderLoader.h"
namespace CubeDemo {

class Shader {
public:
    Shader(const string& vertexPath, const string& fragmentPath);
    ~Shader();
    
    void Use() const;
    void SetMat4(const string& name, const mat4& mat) const;
    void ApplyCamera(const Camera& camera, float aspectRatio) const;
    void SetVec3(const string& name, const vec3& value);
    void SetFloat(const string& name, float value);

private:
    unsigned int ID;
};


}