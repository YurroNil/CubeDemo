// src/rendering/modelLoader.cpp

//验证模型文件
#include "rendering/modelLoader.h"
#include <fstream>
 
void ValidateJson(const json& j) {

    if (!j.contains("meta") || !j["meta"].is_object()) {
        throw std::runtime_error("JSON无效：缺少meta部分");
    }
    
    if (!j.contains("shaders") || !j["shaders"].is_object()) {
        throw std::runtime_error("JSON无效：缺少shaders部分");
    }
    
    if (!j.contains("uniforms") || !j["uniforms"].is_object()) {
        throw std::runtime_error("JSON无效：缺少uniforms部分");
    }
    
    // 验证shaders结构
    const auto& shaders = j["shaders"];
    if (!shaders.contains("vertex") || !shaders["vertex"].is_string()) {
        throw std::runtime_error("JSON无效：缺少vertex着色器路径");
    }
    
    if (!shaders.contains("fragment") || !shaders["fragment"].is_string()) {
        throw std::runtime_error("JSON无效：缺少fragment着色器路径");
    }
    
    // 验证uniforms结构
    const auto& unif = j["uniforms"];
    if (!unif.contains("vertex_size") || !unif["vertex_size"].is_number_integer()) {
        throw std::runtime_error("JSON无效：缺少vertex_size定义");
    }
    
    if (!unif.contains("vertices") || !unif["vertices"].is_array()) {
        throw std::runtime_error("JSON无效：缺少vertices数组");
    }
    
    if (!unif.contains("normals") || !unif["normals"].is_array()) {
        throw std::runtime_error("JSON无效：缺少normals数组");
    }
    
    if (!unif.contains("material") || !unif["material"].is_object()) {
        throw std::runtime_error("JSON无效：缺少material定义");
    }
    
    // 验证material结构
    const auto& material = unif["material"];
    if (!material.contains("ambient") || !material["ambient"].is_array() || material["ambient"].size() != 3) {
        throw std::runtime_error("JSON无效：无效的ambient材质属性");
    }
    
    if (!material.contains("diffuse") || !material["diffuse"].is_array() || material["diffuse"].size() != 3) {
        throw std::runtime_error("JSON无效：无效的diffuse材质属性");
    }
    
    if (!material.contains("specular") || !material["specular"].is_array() || material["specular"].size() != 3) {
        throw std::runtime_error("JSON无效：无效的specular材质属性");
    }
    
    if (!material.contains("shininess") || !material["shininess"].is_number_float()) {
        throw std::runtime_error("JSON无效：缺少shininess定义");
    }
}

ModelData ModelLoader::LoadFromJson(const string& filePath) {
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
        throw std::runtime_error("JSON粘贴错误：" + string(e.what()));
    }

    //解析
    ModelData model;
    
    // 元数据（默认值unnamed）
    model.name = data["meta"].value("name", "unnamed");
 
    // 着色器（增加存在性检查）
    model.shaders.vertexShader = data["shaders"]["vertex"].get<string>();
    model.shaders.fragmentShader = data["shaders"]["fragment"].get<string>();
 
    // 顶点数据（增加类型验证）
    const auto& unif = data["uniforms"];
    model.vertexComponents = unif["vertex_size"].get<int>();
    
    // 使用get_to验证数组类型
    model.vertices = unif["vertices"].get<std::vector<float>>();
    model.normals = unif["normals"].get<std::vector<float>>();
 
    // 材质数据（增加索引检查）
    const auto& material = unif["material"];
    model.material.ambient = vec3(
        material["ambient"][0].get<float>(),
        material["ambient"][1].get<float>(),
        material["ambient"][2].get<float>()
    );
    
    model.material.diffuse = vec3(
        material["diffuse"][0].get<float>(),
        material["diffuse"][1].get<float>(),
        material["diffuse"][2].get<float>()
    );
    
    model.material.specular = vec3(
        material["specular"][0].get<float>(),
        material["specular"][1].get<float>(),
        material["specular"][2].get<float>()
    );
    
    model.material.shininess = material["shininess"].get<float>();

    return model;
}
