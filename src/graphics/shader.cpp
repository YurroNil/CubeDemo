// renderer/shader.cpp

#include "graphics/shader.h"
#include "utils/glfwKits.h"
#include "utils/streams.h"
using ifs = std::ifstream;

namespace CubeDemo {

string Shader::Load(const string& path) {
    string code;
    ifs file;
    
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
Shader::Shader(const string& vertexPath, const string& fragmentPath) {

    // 加载着色器
    string vertexCode = Load(vertexPath);
    string fragmentCode = Load(fragmentPath);
    
    // 编译顶点着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vShaderCode = vertexCode.c_str();
    glShaderSource(vertexShader, 1, &vShaderCode, NULL);
    glCompileShader(vertexShader);

    // 编译片段着色器
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fShaderCode = fragmentCode.c_str();
    glShaderSource(fragmentShader, 1, &fShaderCode, NULL);
    glCompileShader(fragmentShader);

    // 创建着色器程序
    ID = glCreateProgram();
    glAttachShader(ID, vertexShader);
    glAttachShader(ID, fragmentShader);
    glLinkProgram(ID);

    // 删除着色器对象
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
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

void Shader::ApplyCamera(const Camera& camera, float aspectRatio) const {
    mat4 projection = glm::perspective(
        glm::radians(camera.Zoom),
        aspectRatio,    // 使用宽高比而不是固定分辨率
        0.1f, 100.0f
    );
    SetMat4("projection", projection);
    SetMat4("view", camera.GetViewMatrix());
}

}