// include/prefabs/lights/data.h
#pragma once
#include "prefabs/lights/fwd.h"

namespace CubeDemo::Prefabs {

// 方向光（如太阳）
struct DirLight {
    string name = "方向光", id = "dir_light", type;
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float sourceRadius = 0.5f;      // 光源半径（柔化阴影）
    float sourceSoftness = 0.2f;    // 光源柔和度
    vec3 skyColor = vec3(0.4, 0.6, 0.8); // 天光颜色
    float atmosphereThickness = 0.5f; // 大气厚度

    void SetShader(Shader& shader);
};

// 点光源（如灯泡）
struct PointLight {
    string name = "点光", id = "point_light", type;
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
    string name = "聚光", id = "spot_light", type;

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

struct SkyLight {
    string name = "天空光", id = "sky_light", type;
    vec3 color;
    float intensity;
    float horizonBlend;

    float groundReflection = 0.3f;  // 地面反射强度
    float cloudOpacity = 0.4f;      // 云层不透明度
    vec3 cloudColor = vec3(0.9, 0.9, 0.95); // 云层颜色
    
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
