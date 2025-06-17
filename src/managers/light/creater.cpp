// src/managers/light/creater.cpp
#include "pch.h"
#include "managers/light/creater.h"
#include "prefabs/lights/data.h"

namespace CubeDemo::Managers {

LightCreater::LightCreater() {
}

// 方向光
DL* LightCreater::DirLight() {
    // 创建太阳
    return new DL {
        .direction = vec3(-0.5f, -1.0f, -0.3f),
        .ambient = vec3(0.2f),
        .diffuse = vec3(0.5f),
        .specular = vec3(0.4f)
    };
}

// 点光源
PL* LightCreater::PointLight() {
    return new PL();
}

// 聚光
SL* LightCreater::SpotLight() {
    return new SL();
}

}   // namespace CubeDemo::Managers
