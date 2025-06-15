// src/managers/light/getter.cpp
#include "pch.h"
#include "managers/light/getter.h"

namespace CubeDemo::Managers {

// Getters
DL* LightGetter::DirLight() const { return m_DirLight; }
PL* LightGetter::PointLight() const { return m_PointLight; }
SL* LightGetter::SpotLight() const { return m_SpotLight; }

// Setters
LightGetter& LightGetter::SetDirLight(DL* ptr) {
    m_DirLight = ptr; return *this;
}
LightGetter& LightGetter::SetPointLight(PL* ptr) {
    m_PointLight = ptr; return *this;
}
LightGetter& LightGetter::SetSpotLight(SL* ptr) {
    m_SpotLight = ptr; return *this;
}
}   // namespace CubeDemo::Managers
