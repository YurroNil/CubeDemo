// src/scenes/dynamic_scene.cpp
#include "pch.h"
#include "scenes/dynamic_scene.h"

namespace CubeDemo::Scenes {

DynamicScene::DynamicScene(const SceneInfo& info) 
    : m_info(info) {
    m_id = info.id;
    m_name = info.name;
}
} // namespace CubeDemo::Scenes