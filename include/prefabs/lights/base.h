// include/prefabs/lights/base.h
#pragma once
#include "kits/glm.h"

namespace CubeDemo::Prefabs {

// 平行光（如太阳）
struct DirLight {
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
using DL = CubeDemo::Prefabs::DirLight;

// 点光源（如灯泡）
struct PointLight {
    vec3 position;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;  // 衰减系数
    float linear;
    float quadratic;
};
using PL = CubeDemo::Prefabs::PointLight;

// 聚光灯
struct SpotLight {

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
};
using SL = CubeDemo::Prefabs::SpotLight;
} // namespace CubeDemo::Prefabs
