// src/scenes/scene_info.cpp
#include "pch.h"
#include "scenes/base.h"

namespace CubeDemo::Scenes {

SceneInfo SceneInfo::FromJson(const json& j) {
    SceneInfo info;
    
    // 解析基本字段
    info.id = j.value("id", "unknown");
    info.name = j.value("name", "未命名场景");
    info.description = j.value("description", "");
    info.author = j.value("author", "");
    info.icon = j.value("icon", "");
    
    // 解析场景配置
    if (j.contains("scene_config")) {
        const auto& sceneConfig = j["scene_config"];
        for (const auto& [type, path] : sceneConfig.items()) {
            info.prefabs.push_back({type, path.get<string>()});
        }
    }
    
    return info;
}
} // namespace CubeDemo::Scenes
