// src/managers/lightcpp
#include "pch.h"
#include "managers/lightMng.h"
#include "prefabs/lights/volum_beam.h"
#include "graphics/shader.h"

// 别名
using Shader = CubeDemo::Shader;

namespace CubeDemo {
namespace Managers {

LightMng::LightMng() {
    // 显式初始化成员
    Create = LightCreater();
}

// 创建场景管理器
LightMng* LightMng::CreateInst() {
    if(m_InstCount > 0) {
        std::cerr << "[LightMng] 光源创建失败，因为当前光源管理器数量为: " << m_InstCount << std::endl;
        return nullptr;
    }
    m_InstCount++;
    return new LightMng();
}
// 显式实例化模板
template void LightMng::SetLightsData<DL>(const string&, DL*);
template void LightMng::SetLightsData<SL>(const string&, SL*);
template void LightMng::SetLightsData<DL, SL>(const string&, DL*, SL*);
template void LightMng::SetLightsData<SL, DL>(const string&, SL*, DL*);
template void LightMng::SetLightsData<DL, PL>(const string&, DL*, PL*);
template void LightMng::SetLightsData<DL, SkL>(const string&, DL*, SkL*);
template void LightMng::SetLightsData<DL, PL, SkL>(const string&, DL*, PL*, SkL*);
template void LightMng::SetLightsData<Prefabs::VolumBeam>(const string&, Prefabs::VolumBeam*);
}   // namespace Managers


namespace Prefabs {

/* ---------设置着色器--------- */
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
void SL::SetShader(Shader& shader) {
    const string temp_id = "spot_light";

    shader.SetVec3(temp_id + ".position", position);
    shader.SetVec3(temp_id + ".direction", direction);
    shader.SetFloat(temp_id + ".cutOff", glm::cos(glm::radians(cutOff)));
    shader.SetFloat(temp_id + ".outerCutOff", glm::cos(glm::radians(outerCutOff)));

    shader.SetVec3(temp_id + ".ambient", ambient);
    shader.SetVec3(temp_id + ".diffuse", diffuse);
    shader.SetVec3(temp_id + ".specular", specular);

    shader.SetFloat(temp_id + ".constant", constant);
    shader.SetFloat(temp_id + ".linear", linear);
    shader.SetFloat(temp_id + ".quadratic", quadratic);
}
// 点光
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
void SkL::SetShader(Shader& shader) {
    const string temp_id = "sky_light";

    shader.SetVec3(temp_id + ".color", color);
    shader.SetFloat(temp_id + ".intensity", intensity);
    shader.SetFloat(temp_id + ".horizonBlend", horizonBlend);
    
    shader.SetFloat(temp_id + ".groundReflection", groundReflection);
    shader.SetFloat(temp_id + ".cloudOpacity", cloudOpacity);
    shader.SetVec3(temp_id + ".cloudColor", cloudColor);
}

}}   // namespace
