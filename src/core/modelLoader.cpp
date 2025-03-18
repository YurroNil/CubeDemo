#include "rendering/modelLoader.h"
#include <fstream>

using json = nlohmann::json;

//验证模型文件
void ModelLoader::ValidateJson(const json& j) {
    if (!j.contains("uniforms") || !j["uniforms"].is_object()) {
        throw std::runtime_error("JSON无效：缺少uniforms部分");
    }
}

ModelData ModelLoader::LoadFromJson(const std::string& filePath) {
    //获取
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("打开模型文件失败：" + filePath);
    }

    json data;
    try {
        file >> data;
        ValidateJson(data);
    } catch (const json::exception& e) {
        throw std::runtime_error("JSON粘贴错误：" + std::string(e.what()));
    }

    //解析

    ModelData model;
    // 解析元数据
    model.name = data["meta"].value("name", "unnamed");
    
    // 解析着色器
    model.shaders.vertexShader = data["shaders"]["vertex"].get<std::string>();
    model.shaders.fragmentShader = data["shaders"]["fragment"].get<std::string>();

    // 解析顶点数据
    const auto& unif = data["uniforms"];
    model.vertexComponents = unif["vertex_size"].get<int>();
    model.vertices = unif["vertices"].get< std::vector<float> >();

    return model;
}
