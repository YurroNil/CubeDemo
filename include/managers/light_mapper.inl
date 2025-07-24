// include/managers/light_mapper.inl
#pragma once
#include "prefabs/lights.h"

namespace CubeDemo::Prefabs::Lights::Mapper {

// 基础映射模板 - 使用类型特化确保安全
template<typename T>
void MapLightData(const json& j, T& data) {
    // 通用属性（除BeamEffects外的所有光源共有）
    if constexpr (!std::is_same_v<T, BeamEffects>) {
        if (j.contains("name")) data.name = j["name"].get<string>();
        if (j.contains("id")) data.id = j["id"].get<string>();
        if (j.contains("type")) data.type = j["type"].get<string>();
        if (j.contains("icon_path")) data.iconPath = j["icon_path"].get<string>();
        if (j.contains("description")) data.description = j["description"].get<string>();
        
        // 位置和方向
        if (j.contains("position")) {
            auto pos = j["position"];
            data.position = vec3(pos[0], pos[1], pos[2]);
        }
        
        if (j.contains("direction")) {
            auto dir = j["direction"];
            data.direction = vec3(dir[0], dir[1], dir[2]);
        }
    }
    
    // 类型特定属性
    if constexpr (std::is_same_v<T, DL>) {
        if (j.contains("ambient")) {
            auto arr = j["ambient"];
            data.ambient = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("diffuse")) {
            auto arr = j["diffuse"];
            data.diffuse = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("specular")) {
            auto arr = j["specular"];
            data.specular = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("skyColor")) {
            auto arr = j["skyColor"];
            data.skyColor = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("sourceRadius")) data.sourceRadius = j["sourceRadius"].get<float>();
        if (j.contains("sourceSoftness")) data.sourceSoftness = j["sourceSoftness"].get<float>();
        if (j.contains("atmosphereThickness")) data.atmosphereThickness = j["atmosphereThickness"].get<float>();
    }
    else if constexpr (std::is_same_v<T, PL>) {
        if (j.contains("ambient")) {
            auto arr = j["ambient"];
            data.ambient = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("diffuse")) {
            auto arr = j["diffuse"];
            data.diffuse = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("specular")) {
            auto arr = j["specular"];
            data.specular = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("constant")) data.constant = j["constant"].get<float>();
        if (j.contains("linear")) data.linear = j["linear"].get<float>();
        if (j.contains("quadratic")) data.quadratic = j["quadratic"].get<float>();
    }
    else if constexpr (std::is_same_v<T, SL>) {
        if (j.contains("ambient")) {
            auto arr = j["ambient"];
            data.ambient = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("diffuse")) {
            auto arr = j["diffuse"];
            data.diffuse = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("specular")) {
            auto arr = j["specular"];
            data.specular = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("constant")) data.constant = j["constant"].get<float>();
        if (j.contains("linear")) data.linear = j["linear"].get<float>();
        if (j.contains("quadratic")) data.quadratic = j["quadratic"].get<float>();
        if (j.contains("cutOff")) data.cutOff = j["cutOff"].get<float>();
        if (j.contains("outerCutOff")) data.outerCutOff = j["outerCutOff"].get<float>();
    }
    else if constexpr (std::is_same_v<T, SkL>) {
        if (j.contains("color")) {
            auto arr = j["color"];
            data.color = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("cloudColor")) {
            auto arr = j["cloudColor"];
            data.cloudColor = vec3(arr[0], arr[1], arr[2]);
        }
        if (j.contains("intensity")) data.intensity = j["intensity"].get<float>();
        if (j.contains("horizonBlend")) data.horizonBlend = j["horizonBlend"].get<float>();
        if (j.contains("groundReflection")) data.groundReflection = j["groundReflection"].get<float>();
        if (j.contains("cloudOpacity")) data.cloudOpacity = j["cloudOpacity"].get<float>();
    }
    else if constexpr (std::is_same_v<T, BeamEffects>) {
        // 只映射 BeamEffects 实际拥有的属性
        if (j.contains("name")) data.name = j["name"].get<string>();
        if (j.contains("id")) data.id = j["id"].get<string>();
        if (j.contains("type")) data.type = j["type"].get<string>();
        if (j.contains("description")) data.description = j["description"].get<string>();
        if (j.contains("noiseTexture")) data.noiseTexture = j["noiseTexture"].get<string>();

        if (j.contains("radius")) data.radius = j["radius"].get<float>();
        if (j.contains("height")) data.height = j["height"].get<float>();
        if (j.contains("intensity")) data.intensity = j["intensity"].get<float>();
        if (j.contains("scatterPower")) data.scatterPower = j["scatterPower"].get<float>();
        if (j.contains("alphaMultiplier")) data.alphaMultiplier = j["alphaMultiplier"].get<float>();
        if (j.contains("density")) data.density = j["density"].get<float>();
        if (j.contains("scatterAnisotropy")) data.scatterAnisotropy = j["scatterAnisotropy"].get<float>();
        
        if (j.contains("attenuationFactors")) {
            auto att = j["attenuationFactors"];
            if (att.size() >= 2) {
                data.attenuationFactors.x = att[0].get<float>();
                data.attenuationFactors.y = att[1].get<float>();
            }
        }
    }
}

// 声明显式特化（实现在json_mapper.cpp中）
template<>
void MapLightData<BeamEffects::FlickerParams>(const json& j, BeamEffects::FlickerParams& flicker);

} // namespace CubeDemo::Prefabs::Lights::Mapper
