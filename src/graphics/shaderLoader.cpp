// src/graphics/shaderLoader.cpp

#include "graphics/shaderLoader.h"
using namespace std;

const string ShaderLoader::s_vshPath = "../res/shaders/vertex/core/";
const string ShaderLoader::s_fshPath = "../res/shaders/fragment/core/";


string ShaderLoader::Load(const string& path) {
    string code;
    ifstream file;
    
    file.exceptions(ifstream::failbit | ifstream::badbit);
    
    // 调试
    try {
        file.open(path);
        stringstream stream;
        stream << file.rdbuf(); 
        file.close();
        code = stream.str();
    } catch (ifstream::failure& e) {
        cerr << "[ERROR_SHADER] 文件读取失败" << path << endl;
        cerr << "错误信息: " << e.what() << endl;
    }
    
    return code;
}
