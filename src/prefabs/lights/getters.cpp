// src/prefabs/lights/getters.cpp

#include "prefabs/lights/getters.h"

namespace CubeDemo::Prefabs {

// Getters
DL* LightGetter::DirLight() const { return m_DirLight; }
PL* LightGetter::PointLight() const { return m_PointLight; }
SL* LightGetter::SpotLight() const { return m_SpotLight; }

// Setters
void LightGetter::SetDirLight(DL* ptr) { m_DirLight = ptr; }
void LightGetter::SetPointLight(PL* ptr) { m_PointLight = ptr; }
void LightGetter::SetSpotLight(SL* ptr) { m_SpotLight = ptr; }

}   // namespace CubeDemo::Prefabs
