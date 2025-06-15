// include/prefabs/lights/data.h
#pragma once
#include "prefabs/lights/fwd.h"

namespace CubeDemo::Prefabs {

// 方向光（如太阳）
struct DirLight {
    string name, id, type;
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    void SetShader(Shader& shader);
};

// 点光源（如灯泡）
struct PointLight {
    string name = "dir_light", id, type;
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;  // 衰减系数
    float linear;
    float quadratic;

    void SetShader(Shader& shader);
};

// 聚光
struct SpotLight {
    string name = "spot_light", id, type;

    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
    
    float cutOff;
    float outerCutOff;

    void SetShader(Shader& shader);
};

// 光束效果
struct BeamEffects {
    float radius = 5.0f;
    float height = 30.0f;
    string noiseTexture;
    float intensity = 1.5f;
    float scatterPower = 2.0f;
    vec2 attenuationFactors = vec2(0.1f, 0.05f);
    float alphaMultiplier = 0.6f;
     float density = 0.7f;             // 介质密度
    float scatterAnisotropy = 0.8f;    // 散射各向异性

    struct FlickerParams {
        bool enable = false;
        float min = 0.8f;
        float max = 1.2f;
        float speed = 1.0f;
    };
    
    FlickerParams flicker;
};

} // namespace CubeDemo::Prefabs

#include "managers/light/json_mapper.inl"
