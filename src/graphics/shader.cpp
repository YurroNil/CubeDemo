// src/graphics/shader.cpp
#include "pch.h"

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
Shader::Shader(
    const string& vertex_path, const string& fragment_path,
    const string& geometry_path, const string& compute_path
) {

    if(vertex_path.empty() && fragment_path.empty() && geometry_path.empty() && compute_path.empty()) {
        std::cerr << "[ERROR_SHADER] 着色器路径为空" << std::endl;
        return;
    }
    std::vector<unsigned int> shaders;
    m_ID = glCreateProgram();   // 创建着色器程序

    /* ------------加载着色器------------ */

    if(!vertex_path.empty()) shaders.push_back(InitVertexShader(vertex_path));
    if(!fragment_path.empty()) shaders.push_back(InitFragmentShader(fragment_path));
    if(!geometry_path.empty()) shaders.push_back(InitGeometryShader(geometry_path));
    if(!compute_path.empty()) shaders.push_back(InitComputeShader(compute_path));

    // 链接着色器程序
    glLinkProgram(m_ID);
    for(const auto& shader : shaders) glDeleteShader(shader);
}

// 创建专属着色器
Shader::Shader(const string& path, GLenum type) {

    if(path.empty()) {
        std::cerr << "[ERROR_SHADER] 着色器路径为空" << std::endl;
        return;
    }

    m_ID = glCreateProgram();   // 创建着色器程序

    unsigned int shader =
        type == GL_VERTEX_SHADER   ? InitVertexShader(path)   :
        type == GL_GEOMETRY_SHADER ? InitGeometryShader(path) :
        type == GL_FRAGMENT_SHADER ? InitFragmentShader(path) :
        type == GL_COMPUTE_SHADER  ? InitComputeShader(path)  :
        0;

    if(shader == 0) return;  // 编译失败处理
    
    // 链接着色器程序
    glLinkProgram(m_ID);
    glDeleteShader(shader);
}

unsigned int Shader::CompileShader(GLenum type, const string& source) {
    unsigned int shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    // 错误检查
    GLint success; char infoLog[512];

    string err_type_str =
        type == GL_VERTEX_SHADER   ? "VERTEX"   :
        type == GL_GEOMETRY_SHADER ? "GEOMETRY" :
        type == GL_FRAGMENT_SHADER ? "FRAGMENT" :
        type == GL_COMPUTE_SHADER  ? "COMPUTE"  :
        "UNKNOWN";

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "着色器编译错误 (" << err_type_str << "):\n" << infoLog << std::endl;
        return 0; // 返回0表示编译失败
    }
    return shader;
}

// 初始化顶点着色器程序
int Shader::InitVertexShader(const string& path) {
    if(path.empty()) return -1;
    
    string vert_code = Load(path);
    if(vert_code.empty()) {
        std::cerr << "[ERROR_SHADER] 顶点着色器内容为空: " << path << std::endl;
        return -1;
    }
    unsigned int vert_shader = CompileShader(GL_VERTEX_SHADER, vert_code);  // 正确指定类型
    if(vert_shader == 0) return -1;  // 编译失败处理
    
    glAttachShader(m_ID, vert_shader);
    return vert_shader;
}
// 初始化片段着色器程序
int Shader::InitFragmentShader(const string& path) {
    if(path.empty()) return -1;
    
    string frag_code = Load(path);
    if(frag_code.empty()) {
        std::cerr << "[ERROR_SHADER] 片段着色器内容为空: " << path << std::endl;
        return -1;
    }
    unsigned int frag_shader = CompileShader(GL_FRAGMENT_SHADER, frag_code);  // 正确指定类型
    if(frag_shader == 0) return -1;  // 编译失败处理
    
    glAttachShader(m_ID, frag_shader);
    return frag_shader;
}
// 初始化几何着色器程序
int Shader::InitGeometryShader(const string& path) {
    if(path.empty()) return -1;
    
    string geom_code = Load(path);
    if(geom_code.empty()) {
        std::cerr << "[ERROR_SHADER] 几何着色器内容为空: " << path << std::endl;
        return -1;
    }

    unsigned int geom_shader = CompileShader(GL_GEOMETRY_SHADER, geom_code);  // 正确指定类型
    if(geom_shader == 0) return -1;  // 编译失败处理
    
    glAttachShader(m_ID, geom_shader);
    return geom_shader;
}
// 初始化计算着色器程序
int Shader::InitComputeShader(const string& path) {
    if(path.empty()) return -1;
    
    string comp_code = Load(path);
    if(comp_code.empty()) {
        std::cerr << "[ERROR_SHADER] 计算着色器内容为空: " << path << std::endl;
        return -1;
    }
    unsigned int comp_shader = CompileShader(GL_COMPUTE_SHADER, comp_code);  // 正确指定类型
    if(comp_shader == 0) return -1;  // 编译失败处理
    
    glAttachShader(m_ID, comp_shader);
    return comp_shader;
}

Shader::~Shader() { glDeleteProgram(m_ID); }

void Shader::Use() const { glUseProgram(m_ID); }

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
void Shader::SetVec4(const string& name, const vec4& value) {
    glUniform4fv( glGetUniformLocation(m_ID, name.c_str()), 1, &value[0] );
}
void Shader::SetFloat(const string& name, float value) {
    glUniform1f(glGetUniformLocation(m_ID, name.c_str()), value);
}
void Shader::SetInt(const string& name, int value) const {
    glUniform1i(glGetUniformLocation(m_ID, name.c_str()), value);
}
void Shader::SetBool(const string& name, bool value) {
    glUniform1i(glGetUniformLocation(m_ID, name.c_str()), static_cast<int>(value));
}
}   // namespace CubeDemo
