// include/managers/light/json_mapper.inl
#pragma once

// 别名
using json = nlohmann::json;
using BeamEffects = CubeDemo::Prefabs::BeamEffects;

namespace CubeDemo::Prefabs::Lights::JsonMapper {

// 基础类型映射模板
template <typename T>
struct TypeMapper;

// bool 特化
template <>
struct TypeMapper<bool> {
    static bool Map(const json& j) {
        return j.get<bool>();
    }
};

// int 特化
template <>
struct TypeMapper<int> {
    static int Map(const json& j) {
        return j.get<int>();
    }
};

// float 特化
template <>
struct TypeMapper<float> {
    static float Map(const json& j) {
        return j.get<float>();
    }
};

// string 特化
template <>
struct TypeMapper<string> {
    static string Map(const json& j) {
        return j.get<string>();
    }
};

// vec2映射特化
template <>
struct TypeMapper<vec2> {
    static vec2 Map(const json& j) {
        return {
            j[0].get<float>(),
            j[1].get<float>()
        };
    }
};
// vec3 特化
template <>
struct TypeMapper<vec3> {
    static vec3 Map(const json& j) {
        return {
            j[0].get<float>(),
            j[1].get<float>(),
            j[2].get<float>()
        };
    }
};

// 灯光特性模板
template <typename T>
struct LightTraits;

// BeamEffects特化
template <>
struct LightTraits<BeamEffects> {
    using LightType = BeamEffects;
    
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("radius", &LightType::radius),
        std::make_pair("height", &LightType::height),
        std::make_pair("noiseTexture", &LightType::noiseTexture),
        std::make_pair("intensity", &LightType::intensity),
        std::make_pair("scatterPower", &LightType::scatterPower),
        std::make_pair("alphaMultiplier", &LightType::alphaMultiplier)
    );
};

// 特化flicker映射
template <>
struct LightTraits<BeamEffects::FlickerParams> {
    using LightType = BeamEffects::FlickerParams;
    
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("enable", &LightType::enable),
        std::make_pair("min", &LightType::min),
        std::make_pair("max", &LightType::max),
        std::make_pair("speed", &LightType::speed)
    );
};

// 通用映射函数（支持嵌套结构）
template <typename T>
void MapLightData(const json& config, T& data) {
    using Traits = LightTraits<T>;
    
    std::apply([&](const auto&... pairs) {
        ( (config.contains(pairs.first) ? 
            (data.*(pairs.second) = TypeMapper<
                std::decay_t<decltype(data.*(pairs.second))>
            >::Map(config[pairs.first]), void())
          : void()), ... );
    }, Traits::member_map);
}

// Spot Light特化
template <>
struct LightTraits<SL> {
    using LightType = SL;
    
    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("name",        &LightType::name),
        std::make_pair("id",          &LightType::id),
        std::make_pair("type",        &LightType::type),

        std::make_pair("position",    &LightType::position),
        std::make_pair("direction",   &LightType::direction),
        std::make_pair("ambient",     &LightType::ambient),
        std::make_pair("diffuse",     &LightType::diffuse),
        std::make_pair("specular",    &LightType::specular),

        std::make_pair("constant",    &LightType::constant),
        std::make_pair("linear",      &LightType::linear),
        std::make_pair("quadratic",   &LightType::quadratic),

        std::make_pair("cutOff",      &LightType::cutOff),
        std::make_pair("outerCutOff", &LightType::outerCutOff)
    );
};

// Directional Light特化
template <>
struct LightTraits<DL> {
    using LightType = DL;

    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("name",      &LightType::name),
        std::make_pair("id",        &LightType::id),
        std::make_pair("type",      &LightType::type),

        std::make_pair("position",  &LightType::position),
        std::make_pair("direction", &LightType::direction),
        std::make_pair("ambient",   &LightType::ambient),
        std::make_pair("diffuse",   &LightType::diffuse),
        std::make_pair("specular",  &LightType::specular),

        std::make_pair("sourceRadius",  &LightType::sourceRadius),
        std::make_pair("sourceSoftness", &LightType::sourceSoftness),
        std::make_pair("sourceSoftness",&LightType::sourceSoftness),
        std::make_pair("skyColor",   &LightType::skyColor),
        std::make_pair("atmosphereThickness",  &LightType::atmosphereThickness)
    );
};

// Point Light特化
template <>
struct LightTraits<PL> {
    using LightType = PL;

    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("name",      &LightType::name),
        std::make_pair("id",        &LightType::id),
        std::make_pair("type",      &LightType::type),

        std::make_pair("position",  &LightType::position),
        std::make_pair("direction", &LightType::direction),
        std::make_pair("ambient",   &LightType::ambient),
        std::make_pair("diffuse",   &LightType::diffuse),
        std::make_pair("specular",  &LightType::specular),

        std::make_pair("constant",    &LightType::constant),
        std::make_pair("linear",      &LightType::linear),
        std::make_pair("quadratic",   &LightType::quadratic)
    );
};

// Sky Light特化
template <>
struct LightTraits<SkL> {
    using LightType = SkL;

    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("name",      &LightType::name),
        std::make_pair("id",        &LightType::id),
        std::make_pair("type",      &LightType::type),
        
        std::make_pair("color",  &LightType::color),
        std::make_pair("intensity", &LightType::intensity),
        std::make_pair("horizonBlend", &LightType::horizonBlend),

        std::make_pair("groundReflection", &LightType::groundReflection),
        std::make_pair("cloudOpacity", &LightType::cloudOpacity),
        std::make_pair("cloudColor",   &LightType::cloudColor)
    );
};

} // namespace
