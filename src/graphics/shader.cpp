// src/graphics/shader.cpp
#include "pch.h"
#include "graphics/shader.h"

// 别名
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
    m_ID = glCreateProgram();
    glAttachShader(m_ID, vert_shader);
    glAttachShader(m_ID, frag_shader);
    glLinkProgram(m_ID);

    GLint success;
    char infoLog[512];

    // 顶点着色器编译检查
    glGetShaderiv(vert_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vert_shader, 512, NULL, infoLog);
        std::cerr << "顶点着色器编译失败: " << infoLog << std::endl;
    }

    // 片段着色器编译检查
    glGetShaderiv(frag_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(frag_shader, 512, NULL, infoLog);
        std::cerr << "片段着色器编译失败: " << infoLog << std::endl;
    }

    // 程序链接检查
    glGetProgramiv(m_ID, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_ID, 512, NULL, infoLog);
        std::cerr << "着色器程序链接失败: " << infoLog << std::endl;
    }

    // 删除着色器对象
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
}

Shader::~Shader() {
    glDeleteProgram(m_ID);
}

void Shader::Use() const {
    glUseProgram(m_ID);
}

void Shader::ApplyCamera(const Camera* camera, float aspect) const {
    mat4 projection = glm::perspective(
        glm::radians(camera->attribute.zoom),
        aspect,    // 使用宽高比而不是固定分辨率
        camera->frustumPlane.near_plane,
        camera->frustumPlane.far_plane
    );
    SetMat4("projection", projection);
    SetMat4("view", camera->GetViewMat());
    
}

// 乱七八糟的Setters
void Shader::SetViewPos(const vec3& pos) { SetVec3("viewPos", pos); }
void Shader::SetLightSpaceMat(const mat4& matrix) { SetMat4("lightSpaceMatrix", matrix); }

void Shader::SetMat4(const string& name, const mat4& mat) const {
    glUniformMatrix4fv( glGetUniformLocation(m_ID, name.c_str()), 1, GL_FALSE, &mat[0][0] );
}
void Shader::SetVec2(const string& name, const vec2& value) {
    glUniform2fv( glGetUniformLocation(m_ID, name.c_str()), 1, &value[0] );
}
void Shader::SetVec3(const string& name, const vec3& value) {
    glUniform3fv( glGetUniformLocation(m_ID, name.c_str()), 1, &value[0] );
}
void Shader::SetFloat(const string& name, float value) {
    glUniform1f(glGetUniformLocation(m_ID, name.c_str()), value);
}
void Shader::SetInt(const string& name, int value) const {
    glUniform1i(glGetUniformLocation(m_ID, name.c_str()), value);
}
void Shader::SetBool(const string& name, bool value) {
    glUniform1i(glGetUniformLocation(m_ID, name.c_str()), (int)value);
}
}   // namespace CubeDemo
