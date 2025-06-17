// include/graphics/shader.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo {
class Camera;
class Shader {
public:

    Shader(const string& vertex_path, const string& fragment_path);
    ~Shader();
    
    void Use() const;
    static string Load(const string& path);

    // Setters
    void ApplyCamera(const Camera* camera, float aspect) const;
    void SetMat4(const string& name, const mat4& mat) const;
    void SetVec2(const string& name, const vec2& value);
    void SetVec3(const string& name, const vec3& value);
    void SetFloat(const string& name, float value);
    void SetInt(const string& name, int value) const;
    void SetBool(const string& name, bool value);
    void SetViewPos(const vec3& pos);
    void SetLightSpaceMat(const mat4& matrix);

private:
    unsigned int ID; 
};
}
