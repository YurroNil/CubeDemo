// include/prefabs/lights/cleanners.h
#pragma once
#include "prefabs/lights/getters.h"

namespace CubeDemo::Prefabs {

class LightCleanner : public LightGetter {
public:
    void All();

    void DirLight();
    void PointLight();
    void SpotLight();
};
}   // namespace CubeDemo
