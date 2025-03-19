#pragma once
#include <string>
#include <vector>
#include "json.hpp"
using string = std::string;

struct ModelData {
    string name;
    std::vector<float> vertices;
    // 每个顶点的分量数（如3表示x,y,z）
    int vertexComponents; 
    struct {
        string vertexShader;
        string fragmentShader;
    } shaders;
};

class ModelLoader {
public:
    static ModelData LoadFromJson(const string& filePath);
private:
    static void ValidateJson(const nlohmann::json& j);
};
