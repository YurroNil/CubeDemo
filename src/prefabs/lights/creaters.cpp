// src/prefabs/lights/creaters.cpp

#include "prefabs/lights/creaters.h"
#include <iostream>
namespace CubeDemo::Prefabs {

// 平行光源
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

// 聚光灯
SL* LightCreater::SpotLight() {
    return new SL();
}

}   // namespace CubeDemo
