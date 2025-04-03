// include/resources/modelLoader.h

#pragma once
#include "utils/root.h"
#include "nlohmann/json.hpp"

struct ModelData;
using n_json = nlohmann::json;


class ModelLoader {
public:
    static ModelData* LoadFromJson(const string& filePath);
};

struct ModelData {
    string name;   // 模型名
    std::vector<float> vertices;    // 顶点数据
    int vertexComponents;   // 向量维数

    struct {// 指定的着色器类型
        string vertexShader;
        string fragmentShader;
    } shaders;

    std::vector<float> normals; // 法线数据
    struct {   // 材质数据
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        float shininess;
    } material;


};