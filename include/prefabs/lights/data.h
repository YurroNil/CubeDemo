// include/prefabs/lights/data.h
#pragma once
#include "prefabs/lights/fwd.h"

namespace CubeDemo::Prefabs {

// 方向光（如太阳）
struct DirLight {
    string
        name = "方向光", id = "dir_light",
        type, iconPath, description;

    vec3
        position, direction, ambient, color,
        diffuse, specular, skyColor = vec3(0.4, 0.6, 0.8);

    float
        intensity = 1.0f, sourceRadius = 0.5f, sourceSoftness = 0.2f, atmosphereThickness = 0.5f;

    void Init();
    void SetShader(Shader& shader);
};

// 点光源（如灯泡）
struct PointLight {
    string
        name = "点光", id = "point_light",
        type, iconPath, description;

    vec3 position, direction, ambient, diffuse, specular, color;

    float intensity = 1.0f, constant, linear, quadratic;

    void Init();
    void SetShader(Shader& shader);
};

// 聚光
struct SpotLight {

    string
        name = "聚光", id = "spot_light",
        type, iconPath, description;

    vec3
        position, direction, ambient,
        diffuse, specular, color;
    
    float intensity = 1.0f, constant, linear, quadratic, cutOff, outerCutOff;

    void Init();
    void SetShader(Shader& shader);
};

struct SkyLight {
    string
        name = "天空光", id = "sky_light",
        type, iconPath, description;
    
    vec3 position, direction, color, cloudColor = vec3(0.9, 0.9, 0.95);

    float intensity, horizonBlend, groundReflection = 0.3f, cloudOpacity = 0.4f;

    void Init();
    void SetShader(Shader& shader);
};

// 光束效果
struct BeamEffects {

    string id, type, name, noiseTexture, description;

    float
        radius = 5.0f, height = 30.0f, intensity = 1.5f,
        scatterPower = 2.0f, alphaMultiplier = 0.6f,
        density = 0.7f, scatterAnisotropy = 0.8f;
    
    vec2 attenuationFactors = vec2(0.1f, 0.05f);

    // 体积光束的闪烁参数
    struct FlickerParams {
        bool enable = false;
        float min = 0.8f, max = 1.2f, speed = 1.0f;
    };
    
    FlickerParams flicker;

};

} // namespace CubeDemo::Prefabs

#include "managers/light/json_mapper.inl"
