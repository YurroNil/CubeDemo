// src/managers/light/creater.cpp
#include "pch.h"
#include "managers/light/creater.h"
#include "prefabs/lights/data.h"

namespace CubeDemo::Managers {

LightCreater::LightCreater() {
}

// 方向光
DL* LightCreater::DirLight() {
    return new DL();
}
// 点光源
PL* LightCreater::PointLight() {
    return new PL();
}
// 聚光
SL* LightCreater::SpotLight() {
    return new SL();
}
// 天空光
SkL* LightCreater::SkyLight() {
    return new SkL();
}
}   // namespace CubeDemo::Managers
