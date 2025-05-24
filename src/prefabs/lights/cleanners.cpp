// src/prefabs/lights/cleanners.cpp

#include "prefabs/lights/cleanners.h"

namespace CubeDemo::Prefabs {

// 平行光源

void LightCleanner::All() {
    delete m_DirLight; m_DirLight = nullptr;
    delete m_PointLight; m_PointLight = nullptr;
    delete m_SpotLight; m_SpotLight = nullptr;
}

void LightCleanner::DirLight() {
    delete m_DirLight; m_DirLight = nullptr;
}
// 点光源
void LightCleanner::PointLight() {
    delete m_PointLight; m_PointLight = nullptr;
}
// 聚光灯
void LightCleanner::SpotLight() {
    delete m_SpotLight; m_SpotLight = nullptr;
}

}   // namespace CubeDemo
