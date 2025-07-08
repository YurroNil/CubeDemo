// src/scenes/scene_info.cpp
#include "pch.h"
#include "scenes/base.h"

namespace CubeDemo::Scenes {

SceneInfo SceneInfo::FromJson(const json& j) {
    SceneInfo info;
    
    try {
        // 解析基本字段
        info.id = j.value("id", "unknown");
        info.name = j.value("name", "未命名场景");
        info.description = j.value("description", "");
        info.author = j.value("author", "");
        info.icon = j.value("icon", "");
        info.previewImage = j.value("preview_image", "");
        
        // 解析场景配置
        if (j.contains("scene_config")) {
            const auto& sceneConfig = j["scene_config"];
            for (const auto& [type, path] : sceneConfig.items()) {
                info.prefabs.push_back({type, path.get<string>()});
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "解析SceneInfo失败: " << e.what() << "\nJSON内容: " << j.dump(2) << std::endl;
    }
    
    return info;
}
} // namespace CubeDemo::Scenes
