// include/prefabs/lights/cleanner.h
#pragma once
#include "prefabs/lights/getter.h"

namespace CubeDemo::Prefabs {

class LightCleanner : public LightGetter {
public:
    void All();

    void DirLight();
    void PointLight();
    void SpotLight();
};
}   // namespace CubeDemo
