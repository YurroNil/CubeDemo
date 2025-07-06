// include/managers/light/utils.inl
#pragma once

#include "managers/light/json_mapper.inl"
#include "prefabs/lights/volum_beam.h"

using VolumBeam = CubeDemo::Prefabs::VolumBeam;
using BeamEffects = CubeDemo::Prefabs::BeamEffects;
namespace JsonMapper = CubeDemo::Prefabs::Lights::JsonMapper;

namespace CubeDemo::Managers {

// 核心模板实现 - 处理单个光源配置
template<typename T>
void SetLightDataImpl(const json& light_config, T* light) {
    if constexpr (std::is_same_v<T, DL>) {
        JsonMapper::MapLightData(light_config, *light);
    } 
    else if constexpr (std::is_same_v<T, SL>) {
        JsonMapper::MapLightData(light_config, *light);
    }
    else if constexpr (std::is_same_v<T, VolumBeam>) {
        BeamEffects effects;
        JsonMapper::MapLightData(light_config, effects);
        
        // 特殊处理flicker配置
        if (light_config.contains("flicker")) {
            JsonMapper::MapLightData(light_config["flicker"], effects.flicker);
        }
        
        light->Configure(effects);
    }
    else if constexpr (std::is_same_v<T, PL>) {
        JsonMapper::MapLightData(light_config, *light);
    }
    else if constexpr (std::is_same_v<T, SkL>) {
        JsonMapper::MapLightData(light_config, *light);
    }
}

// 配置读取工具函数
inline json ReadConfig(const string& config_path) {
    return Utils::JsonConfig::GetFileData(config_path);
}

} // namespace CubeDemo::Managers
