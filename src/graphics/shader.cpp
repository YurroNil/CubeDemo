// src/graphics/shader.cpp

#include "graphics/shader.h"
#include "kits/glfw.h"
#include "kits/streams.h"

using ifs = std::ifstream;
namespace CubeDemo {

string Shader::Load(const string& path) {
    string code; ifs file;
    file.exceptions(ifs::failbit | ifs::badbit);
    // 调试
    try {
        file.open(path);
        std::stringstream stream;
        stream << file.rdbuf(); 
        file.close();
        code = stream.str();
    } catch (ifs::failure& e) {
        std::cerr << "[ERROR_SHADER] 文件读取失败" << path << std::endl;
        std::cerr << "错误信息: " << e.what() << std::endl;
    }
    
    return code;
}

//创建着色器program
Shader::Shader(const string& vertex_path, const string& fragment_path) {

    // 加载着色器
    string vertex_code = Load(vertex_path);
    string frag_code = Load(fragment_path);
    
    // 编译顶点着色器
    unsigned int vert_shader = glCreateShader(GL_VERTEX_SHADER);
    const char* vert_shader_code = vertex_code.c_str();
    glShaderSource(vert_shader, 1, &vert_shader_code, NULL);
    glCompileShader(vert_shader);

    // 编译片段着色器
    unsigned int frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* frag_shader_code = frag_code.c_str();
    glShaderSource(frag_shader, 1, &frag_shader_code, NULL);
    glCompileShader(frag_shader);

    // 创建着色器程序
    ID = glCreateProgram();
    glAttachShader(ID, vert_shader);
    glAttachShader(ID, frag_shader);
    glLinkProgram(ID);

    // 删除着色器对象
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
}

Shader::~Shader() {
    glDeleteProgram(ID);
}

void Shader::Use() const {
    glUseProgram(ID);
}

void Shader::SetMat4(const string& name, const mat4& mat) const {
    glUniformMatrix4fv(
        glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]
    );
}

void Shader::SetVec3(const string& name, const vec3& value) {
    glUniform3fv(
        glGetUniformLocation(ID, name.c_str()), 1, &value[0]
    );
}
void Shader::SetFloat(const string& name, float value) {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::SetInt(const string& name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::ApplyCamera(const Camera& camera, float aspect) const {
    mat4 projection = glm::perspective(
        glm::radians(camera.attribute.zoom),
        aspect,    // 使用宽高比而不是固定分辨率
        camera.frustumPlane.near,
        camera.frustumPlane.far
    );
    SetMat4("projection", projection);
    SetMat4("view", camera.GetViewMat());
    
}

void Shader::SetDirLight(const Graphics::DirLight* light) {
    SetVec3("dirLight.direction", light->direction);
    SetVec3("dirLight.ambient", light->ambient);
    SetVec3("dirLight.diffuse", light->diffuse);
    SetVec3("dirLight.specular", light->specular);
}

void Shader::SetViewPos(const vec3& pos) {
    SetVec3("viewPos", pos);
}

void Shader::SetLightSpaceMatrix(const mat4& matrix) {
    SetMat4("lightSpaceMatrix", matrix);
}

}   // namespace CubeDemo
