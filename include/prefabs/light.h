// include/prefabs/light.h
#pragma once
#include "prefabs/lights/creater.h"
#include "prefabs/lights/cleanner.h"

namespace CubeDemo::Prefabs {
class Light {
public:
    LightGetter Get; LightCreater Create; LightCleanner Remove;
};
}
