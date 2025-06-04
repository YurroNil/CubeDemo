// include/prefabs/lights/json_mapper.inl
#pragma once

// 别名
using json = nlohmann::json;

namespace CubeDemo::Prefabs::Lights::JsonMapper {

// 基础类型映射模板
template <typename T>
struct TypeMapper;

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

// 标量类型特化（float）
template <>
struct TypeMapper<float> {
    static float Map(const json& j) {
        return j.get<float>();
    }
};

// 灯光特性模板
template <typename T>
struct LightTraits;

// SpotLight特化
template <>
struct LightTraits<SL> {
    using LightType = SL;
    
    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
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

// DirectionalLight特化
template <>
struct LightTraits<DL> {
    using LightType = DL;

    // JSON键与成员指针的映射
    static constexpr auto member_map = std::make_tuple(
        std::make_pair("position",  &LightType::position),
        std::make_pair("direction", &LightType::direction),
        std::make_pair("ambient",   &LightType::ambient),
        std::make_pair("diffuse",   &LightType::diffuse),
        std::make_pair("specular",  &LightType::specular)
    );
};

// 通用映射函数
template <typename T>
void MapLightData(const json& config, T& light) {
    using Traits = LightTraits<T>;
    
    // 遍历成员映射表
    std::apply(
         [&](const auto&... pairs) {
            // 使用逗号运算符展开参数包，并确保void表达式
            ( (void)(
                (config.contains(pairs.first) ? 
                    (light.*(pairs.second) = TypeMapper<
                        std::decay_t<decltype(light.*(pairs.second))>
                    >::Map(config[pairs.first]), void())  // 逗号运算符确保void类型
                    : void())
            ), ... );  // 折叠表达式展开参数包
        },
        Traits::member_map
    );
}

} // namespace
