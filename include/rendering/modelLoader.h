#pragma once
#include <string>
#include <vector>
#include "3rd-lib/json.hpp"

struct ModelData {
    std::string name;
    std::vector<float> vertices;
    // 每个顶点的分量数（如3表示x,y,z）
    int vertexComponents; 
    struct {
        std::string vertexShader;
        std::string fragmentShader;
    } shaders;
};

class ModelLoader {
public:
    static ModelData LoadFromJson(const std::string& filePath);
private:
    static void ValidateJson(const nlohmann::json& j);
};
