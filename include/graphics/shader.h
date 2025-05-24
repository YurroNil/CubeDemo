// include/graphics/shader.h

#pragma once
#include "core/camera.h"
#include "kits/strings.h"
#include "prefabs/light.h"

// 别名
using DL = CubeDemo::Prefabs::DirLight;
using PL = CubeDemo::Prefabs::PointLight;
using SL = CubeDemo::Prefabs::SpotLight;

namespace CubeDemo {

class Shader {
public:

    Shader(const string& vertex_path, const string& fragment_path);
    ~Shader();
    
    void Use() const;
    static string Load(const string& path);

    // Setters
    void ApplyCamera(const Camera& camera, float aspect) const;
    void SetMat4(const string& name, const mat4& mat) const;
    void SetVec3(const string& name, const vec3& value);
    void SetFloat(const string& name, float value);
    void SetInt(const string& name, int value) const;
    void SetViewPos(const vec3& pos);
    void SetLightSpaceMat(const mat4& matrix);

    void SetDirLight(const string& name, const DL* light);
    void SetSpotLight(const string& name, const SL& light);

private:
    unsigned int ID; 
};
}
