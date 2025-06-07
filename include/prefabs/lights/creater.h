// include/prefabs/lights/creater.h
#pragma once
#include "prefabs/lights/getter.h"

namespace CubeDemo::Prefabs {

class LightCreater : public LightGetter {
public:
    DL* DirLight();
    PL* PointLight();
    SL* SpotLight();
};
}
