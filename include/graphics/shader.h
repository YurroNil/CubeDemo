// include/graphics/shader.h

#pragma once
#include "core/camera.h"
#include "kits/strings.h"
#include "graphics/light.h"

namespace CubeDemo {

class Shader {
public:

    Shader(const string& vertex_path, const string& fragment_path);
    ~Shader();
    
    void Use() const;
    static string Load(const string& path);

    // Setters
    void SetMat4(const string& name, const mat4& mat) const;
    void ApplyCamera(const Camera& camera, float aspect) const;
    void SetVec3(const string& name, const vec3& value);
    void SetFloat(const string& name, float value);
    void SetInt(const string& name, int value) const;
    void SetDirLight(const Graphics::DirLight* light);
    void SetViewPos(const vec3& pos);
    void SetLightSpaceMatrix(const mat4& matrix);

private:
    unsigned int ID; 
};


}