// include/rendering/modelLoader.h

#pragma once
#include <string>
#include <vector>
#include "json.hpp"
#include "glm/glm.hpp"
struct ModelData;
using string = std::string;
using json = nlohmann::json;
using vec3 = glm::vec3;

class ModelLoader {
public:
    ModelData LoadFromJson(const string& filePath);
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