// include/prefabs/lights/fwd.h
#pragma once

namespace CubeDemo::Prefabs {
// 光源数据
struct DirLight; struct PointLight; struct SpotLight;
// 光源管理器
class Light; class LightCreater; class LightCleanner; class LightGetter;
// 阴影
class ShadowMap;
}
// 别名
using DL = CubeDemo::Prefabs::DirLight;
using PL = CubeDemo::Prefabs::PointLight;
using SL = CubeDemo::Prefabs::SpotLight;