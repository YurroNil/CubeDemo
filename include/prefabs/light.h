// include/prefabs/light.h
#pragma once
#include "prefabs/lights/creaters.h"
#include "prefabs/lights/cleanners.h"

namespace CubeDemo::Prefabs {
class Light {
public:
    LightGetter Get; LightCreater Create; LightCleanner Remove;
};
}
