// include/prefabs/lights/getter.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo::Prefabs {

class LightGetter {
public:
    // Getters
    DL* DirLight() const;
    PL* PointLight() const;
    SL* SpotLight() const;

    // Setters
    void SetDirLight(DL* ptr);
    void SetPointLight(PL* ptr);
    void SetSpotLight(SL* ptr);

protected:
    DL* m_DirLight = nullptr;
    PL* m_PointLight = nullptr;
    SL* m_SpotLight = nullptr;
};
}