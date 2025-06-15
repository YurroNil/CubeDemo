// src/managers/lightcpp
#include "pch.h"
#include "managers/lightMng.h"
#include "prefabs/lights/volum_beam.h"
#include "graphics/shader.h"

// 别名
using Shader = CubeDemo::Shader;

namespace CubeDemo {
namespace Managers {

// 创建场景管理器
LightMng* LightMng::CreateInst() {
    if(s_InstCount > 0) {
        std::cerr << "[LightMng] 光源创建失败，因为当前光源管理器数量为: " << s_InstCount << std::endl;
        return nullptr;
    }
    s_InstCount++;
    return new LightMng();
}
// 显式实例化模板
template void LightMng::SetLightsData<DL>(const string&, DL*);
template void LightMng::SetLightsData<SL>(const string&, SL*);
template void LightMng::SetLightsData<DL, SL>(const string&, DL*, SL*);
template void LightMng::SetLightsData<SL, DL>(const string&, SL*, DL*);
template void LightMng::SetLightsData<Prefabs::VolumBeam>(const string&, Prefabs::VolumBeam*);
}   // namespace Managers


namespace Prefabs {

/* ---------设置着色器--------- */
// 方向光
void DL::SetShader(Shader& shader) {

    shader.SetVec3("dir_light.direction", direction);
    shader.SetVec3("dir_light.ambient", ambient);
    shader.SetVec3("dir_light.diffuse", diffuse);
    shader.SetVec3("dir_light.specular", specular);
}
// 聚光
void SL::SetShader(Shader& shader) {

    shader.SetVec3("spot_light.position", position);
    shader.SetVec3("spot_light.direction", direction);
    shader.SetFloat("spot_light.cutOff", glm::cos(glm::radians(cutOff)));
    shader.SetFloat("spot_light.outerCutOff", glm::cos(glm::radians(outerCutOff)));

    shader.SetVec3("spot_light.ambient", ambient);
    shader.SetVec3("spot_light.diffuse", diffuse);
    shader.SetVec3("spot_light.specular", specular);

    shader.SetFloat("spot_light.constant", constant);
    shader.SetFloat("spot_light.linear", linear);
    shader.SetFloat("spot_light.quadratic", quadratic);
}
// 点光
void PL::SetShader(Shader& shader) {
}
}}   // namespace
