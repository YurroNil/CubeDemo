// include/prefabs/lights_fwd.h
#pragma once

namespace CubeDemo::Prefabs {
// 光源预制体
    // 光源数据
    struct DirLight; struct PointLight; struct SpotLight; struct SkyLight;
    // 几何体
    class VolumBeam;
    // 阴影
    class ShadowMap;
}

// 别名
using DL = CubeDemo::Prefabs::DirLight;
using PL = CubeDemo::Prefabs::PointLight;
using SL = CubeDemo::Prefabs::SpotLight;
using SkL = CubeDemo::Prefabs::SkyLight;
using ShadowMap = CubeDemo::Prefabs::ShadowMap;
