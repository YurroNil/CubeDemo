// include/managers/light/getter.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo::Managers {

class LightGetter {
public:
    // Getters
    DL* DirLight() const;
    PL* PointLight() const;
    SL* SpotLight() const;

    // Setters
    LightGetter& SetDirLight(DL* ptr);
    LightGetter& SetPointLight(PL* ptr);
    LightGetter& SetSpotLight(SL* ptr);

protected:
    DL* m_DirLight = nullptr;
    PL* m_PointLight = nullptr;
    SL* m_SpotLight = nullptr;
};
}