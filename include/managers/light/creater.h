// include/managers/light/creater.h
#pragma once
#include "prefabs/lights/data.h"

namespace CubeDemo::Managers {

class LightCreater {
public:
    LightCreater();

    DL* DirLight(); PL* PointLight();
    SL* SpotLight(); SkL* SkyLight();
};
}

