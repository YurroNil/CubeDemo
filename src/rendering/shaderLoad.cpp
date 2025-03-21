#include "rendering/shaderLoader.h"
using namespace std;


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
    }
    
    return code;
}
