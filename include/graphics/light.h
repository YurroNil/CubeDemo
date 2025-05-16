// include/graphics/light.h
#pragma once
#include "kits/glm.h"

namespace CubeDemo::Graphics {

// 平行光（如太阳）
struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

// 点光源（如灯泡）
struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;  // 衰减系数
    float linear;
    float quadratic;
};

} // namespace CubeDemo::Graphics