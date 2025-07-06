// src/managers/light/json_mapper.cpp
#include "pch.h"
#include "managers/light/json_mapper.inl"

namespace CubeDemo::Prefabs::Lights::JsonMapper {

// 显式特化的实现
template<>
void MapLightData<BeamEffects::FlickerParams>(const json& j, BeamEffects::FlickerParams& flicker) {
    if (j.contains("enable")) flicker.enable = j["enable"].get<bool>();
    if (j.contains("min")) flicker.min = j["min"].get<float>();
    if (j.contains("max")) flicker.max = j["max"].get<float>();
    if (j.contains("speed")) flicker.speed = j["speed"].get<float>();
}

// 显式实例化需要的模板
template void MapLightData<DL>(const json&, DL&);
template void MapLightData<PL>(const json&, PL&);
template void MapLightData<SL>(const json&, SL&);
template void MapLightData<SkL>(const json&, SkL&);
template void MapLightData<BeamEffects>(const json&, BeamEffects&);

} // namespace CubeDemo::Prefabs::Lights::JsonMapper
