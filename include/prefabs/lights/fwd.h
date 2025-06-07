// include/prefabs/lights/fwd.h
#pragma once

namespace CubeDemo::Prefabs {

struct DirLight; struct PointLight; struct SpotLight;

class Light; class ShadowMap;

}
using DL = CubeDemo::Prefabs::DirLight;
using PL = CubeDemo::Prefabs::PointLight;
using SL = CubeDemo::Prefabs::SpotLight;