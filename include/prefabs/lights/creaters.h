// include/prefabs/lights/creaters.h
#pragma once
#include "prefabs/lights/getters.h"

namespace CubeDemo::Prefabs {

class LightCreater : public LightGetter {
public:
    DL* DirLight();
    PL* PointLight();
    SL* SpotLight();
};
}
