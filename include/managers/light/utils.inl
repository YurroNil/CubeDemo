// include/managers/light.inl
#pragma once
// 别名
using VolumBeam = CubeDemo::Prefabs::VolumBeam;
namespace JsonMapper = CubeDemo::Prefabs::Lights::JsonMapper;

namespace CubeDemo::Managers {

// 核心模板实现（处理单个光源）
template<typename T>
void SetLightsDataImpl(const json& config, T* light) {
    if constexpr (std::is_same_v<T, DL>) {
        JsonMapper::MapLightData(config["LightArgs"]["DirLight"], *light);
    } 
    else if constexpr (std::is_same_v<T, SL>) {
        JsonMapper::MapLightData(config["LightArgs"]["SpotLight"], *light);
    }
    else if constexpr (std::is_same_v<T, VolumBeam>) {
        BeamEffects effects;
        const auto& key = config["LightArgs"]["BeamEffects"];

        JsonMapper::MapLightData(key, effects);
        // 嵌套配置flicker
        if (key.contains("flicker")) {
            JsonMapper::MapLightData(
                key["flicker"],
                effects.flicker
            );
        }
        light->Configure(effects);
    }
}

// 配置读取工具函数
inline json ReadConfig(const string& config_path) {
    return Utils::JsonConfig::GetFileData(config_path);
}

// 模板展开函数（处理多个光源）
template<typename T, typename... Args>
void LightMng::SetLightsData(const string& config_path, T* first, Args*... args) {
    json config = ReadConfig(config_path);
    SetLightsDataImpl(config, first);
    
    // 递归展开参数包
    if constexpr (sizeof...(Args) > 0) {
        SetLightsData(config_path, args...);
    }
}
extern template void LightMng::SetLightsData<DL>(const string&, DL*);
extern template void LightMng::SetLightsData<SL>(const string&, SL*);
extern template void LightMng::SetLightsData<DL, SL>(const string&, DL*, SL*);
extern template void LightMng::SetLightsData<SL, DL>(const string&, SL*, DL*);
extern template void LightMng::SetLightsData<VolumBeam>(const string&, VolumBeam*);

}   // namespace CubeDemo::Managers
