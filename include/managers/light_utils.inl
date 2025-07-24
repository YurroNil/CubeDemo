// include/managers/light_utils.inl
#pragma once

#include "managers/light_mapper.inl"
#include "prefabs/volum_beam.h"

using VolumBeam = CubeDemo::Prefabs::VolumBeam;
using BeamEffects = CubeDemo::Prefabs::BeamEffects;
namespace LightMapper = CubeDemo::Prefabs::Lights::Mapper;

namespace CubeDemo::Managers {

// 核心模板实现 - 处理单个光源配置
template<typename T>
void SetLightDataImpl(const json& light_config, T* light) {
    if constexpr (std::is_same_v<T, DL>) {
        LightMapper::MapLightData(light_config, *light);
    } 
    else if constexpr (std::is_same_v<T, SL>) {
        LightMapper::MapLightData(light_config, *light);
    }
    else if constexpr (std::is_same_v<T, VolumBeam>) {
        BeamEffects effects;
        LightMapper::MapLightData(light_config, effects);
        
        // 特殊处理flicker配置
        if (light_config.contains("flicker")) {
            LightMapper::MapLightData(light_config["flicker"], effects.flicker);
        }
        
        light->Configure(effects);
    }
    else if constexpr (std::is_same_v<T, PL>) {
        LightMapper::MapLightData(light_config, *light);
    }
    else if constexpr (std::is_same_v<T, SkL>) {
        LightMapper::MapLightData(light_config, *light);
    }
}

// 配置读取工具函数
inline json ReadConfig(const string& config_path) {
    return Utils::JsonConfig::GetFileData(config_path);
}
} // namespace CubeDemo::Managers
