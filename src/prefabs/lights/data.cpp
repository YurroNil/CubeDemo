// src/prefabs/lights/data.cpp
#include "pch.h"
#include "prefabs/lights/data.h"

namespace CubeDemo::Prefabs {

/* ---------设置着色器--------- */

void DL::Init() {}

// 方向光
void DL::SetShader(Shader& shader) {
    const string temp_id = "dir_light";

    shader.SetVec3(temp_id + ".direction", direction);
    shader.SetVec3(temp_id + ".ambient", ambient);
    shader.SetVec3(temp_id + ".diffuse", diffuse);
    shader.SetVec3(temp_id + ".specular", specular);

    shader.SetFloat(temp_id + ".sourceRadius", sourceRadius);
    shader.SetFloat(temp_id + ".sourceSoftness", sourceSoftness);
    shader.SetVec3(temp_id + ".skyColor", skyColor);
    shader.SetFloat(temp_id + ".atmosphereThickness", atmosphereThickness);
}

// 聚光
void SL::Init() {}

void SL::SetShader(Shader& shader) {
    using namespace glm;

    const string temp_id = "spot_light";

    shader.SetVec3(temp_id + ".position", position);
    shader.SetVec3(temp_id + ".direction", direction);
    shader.SetFloat(temp_id + ".cutOff", cos(radians(cutOff)));
    shader.SetFloat(temp_id + ".outerCutOff", cos(radians(outerCutOff)));

    shader.SetVec3(temp_id + ".ambient", ambient);
    shader.SetVec3(temp_id + ".diffuse", diffuse);
    shader.SetVec3(temp_id + ".specular", specular);

    shader.SetFloat(temp_id + ".constant", constant);
    shader.SetFloat(temp_id + ".linear", linear);
    shader.SetFloat(temp_id + ".quadratic", quadratic);
}
// 点光
void PL::Init() {}

void PL::SetShader(Shader& shader) {
    const string temp_id = "point_light";

    shader.SetVec3(temp_id + ".position", position);
    shader.SetVec3(temp_id + ".direction", direction);
    shader.SetVec3(temp_id + ".ambient", ambient);
    shader.SetVec3(temp_id + ".diffuse", diffuse);
    shader.SetVec3(temp_id + ".specular", specular);

    shader.SetFloat(temp_id + ".constant", constant);
    shader.SetFloat(temp_id + ".linear", linear);
    shader.SetFloat(temp_id + ".quadratic", quadratic);
}
// 天空光
void SkL::Init() {}

void SkL::SetShader(Shader& shader) {
    const string temp_id = "sky_light";

    shader.SetVec3(temp_id + ".color", color);
    shader.SetFloat(temp_id + ".intensity", intensity);
    shader.SetFloat(temp_id + ".horizonBlend", horizonBlend);
    
    shader.SetFloat(temp_id + ".groundReflection", groundReflection);
    shader.SetFloat(temp_id + ".cloudOpacity", cloudOpacity);
    shader.SetVec3(temp_id + ".cloudColor", cloudColor);
}
} // namespace
