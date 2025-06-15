// include/managers/light/cleanner.h
#pragma once
#include "managers/light/getter.h"

namespace CubeDemo::Managers {

class LightCleanner : public LightGetter {
public:
    void All();

    LightCleanner& DirLight();
    LightCleanner& PointLight();
    LightCleanner& SpotLight();
};
}   // namespace CubeDemo
