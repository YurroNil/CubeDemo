// include/rendering/modelLoader.h

#pragma once
#include <string>
#include <vector>
#include "json.hpp"

struct ModelData;

class ModelLoader {
public:
    ModelData LoadFromJson(const std::string& filePath);
};

struct ModelData {
    std::string name;
    std::vector<float> vertices;
    int vertexComponents;
    struct {
        std::string vertexShader;
        std::string fragmentShader;
    } shaders;
};