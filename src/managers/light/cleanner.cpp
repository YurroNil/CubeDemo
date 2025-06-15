// src/managers/light/cleanner.cpp
#include "pch.h"
#include "managers/light/cleanner.h"
#include "prefabs/lights/volum_beam.h"
#include "graphics/shader.h"

// 别名
using VolumBeam = CubeDemo::Prefabs::VolumBeam;
using VBR = CubeDemo::Prefabs::VolumBeam::Remover;

namespace CubeDemo {
namespace Managers {

/* ----------LightCleanners---------- */

// 方向光
void LightCleanner::All() {
    delete m_DirLight; m_DirLight = nullptr;
    delete m_PointLight; m_PointLight = nullptr;
    delete m_SpotLight; m_SpotLight = nullptr;
}

LightCleanner& LightCleanner::DirLight() {
    delete m_DirLight; m_DirLight = nullptr;

    return *this;
}
// 点光源
LightCleanner& LightCleanner::PointLight() {
    delete m_PointLight; m_PointLight = nullptr;

    return *this;
}
// 聚光
LightCleanner& LightCleanner::SpotLight() {
    delete m_SpotLight; m_SpotLight = nullptr;

    return *this;
}
}  // namespace Managers

/* ----------VolumBeam.Remover---------- */
namespace Prefabs {

// 通过该构造函数，让VolumBeam.Remover自动绑定到VolumBeam
VolumBeam::VolumBeam()
    : Remove(this) {}
// 通过该构造函数，自动绑定VolumBeam.Remover的主人(即VolumBeam类实例)
VBR::Remover(VolumBeam* owner)
    : m_owner(owner) {}

VBR& VBR::VolumShader() {
    delete m_owner->m_VolumShader; m_owner->m_VolumShader = nullptr;
    return *this;
}
VBR& VBR::LightCone() {
    delete m_owner->m_LightVolume; m_owner->m_LightVolume = nullptr;
    return *this;
}
VBR& VBR::NoiseTexture() {
    m_owner->m_noiseTexture = nullptr;
    return *this;
}
}}   // namespace
