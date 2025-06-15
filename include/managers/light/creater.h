// include/managers/light/creater.h
#pragma once
#include "managers/light/getter.h"

namespace CubeDemo::Managers {

class LightCreater : public LightGetter {
public:
    DL* DirLight();
    PL* PointLight();
    SL* SpotLight();
};
}
