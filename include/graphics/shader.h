// include/graphics/shader.h
#pragma once

namespace CubeDemo {
class Camera;
class Shader {
public:

    Shader(
        const string& vertex_path = "",
        const string& fragment_path = "",
        const string& geometry_path = "",
        const string& compute_path = ""
    );
    Shader(const string& compute_path, GLenum type);
    ~Shader();
    unsigned int CompileShader(GLenum type, const string& source);
    void Use() const;
    static string Load(const string& path);

    // Initers
    int InitVertexShader(const string& path);
    int InitFragmentShader(const string& path);
    int InitGeometryShader(const string& path);
    int InitComputeShader(const string& path);

    // Setters
    void ApplyCamera(const Camera* camera, float aspect) const;
    void SetMat4(const string& name, const mat4& mat) const;
    void SetVec2(const string& name, const vec2& value);
    void SetVec3(const string& name, const vec3& value);
    void SetVec4(const string& name, const vec4& value);
    void SetFloat(const string& name, float value);
    void SetInt(const string& name, int value) const;
    void SetBool(const string& name, bool value);
    void SetViewPos(const vec3& pos);
    void SetLightSpaceMat(const mat4& matrix);
    unsigned int GetID() const { return m_ID; }

private:
    unsigned int m_ID; 
};
}
