// src/scenes/sceneMng.cpp
#include "scenes/sceneMng.h"

namespace CubeDemo::Scenes {

// 别名
namespace Internal = CubeDemo::Scenes::Internal;

// 初始化
void SceneMng::Init(Light& light) {

    auto& registry = Internal::GetSceneRegistry(*this);
    if (auto it = registry.find(Current); it != registry.end()) {
        it->second.instance.get().Init(*this, light);
    }
}

// 清理所有场景
void SceneMng::CleanAllScenes(Light& light) {
    CleanScene<SceneID::DEFAULT>(light);
    CleanScene<SceneID::NIGHT>(light);
    // 扩展时添加新场景清理
}

// 切换场景
void SceneMng::SwitchTo(SceneID target, Light& light) {
    // 清理旧场景（使用运行时多态）
    auto& registry = Internal::GetSceneRegistry(*this);
    if (auto it = registry.find(Current); it != registry.end()) {
        it->second.instance.get().Cleanup(light);
    }
 
    Current = target;
    Init(light);
}

// 渲染
void SceneMng::Rendering(
    SceneID current,       // 场景ID
    GLFWwindow* window,    // 窗口
    Camera* camera,        // 摄像机
    const Light& light,    // 光源
    ShadowMap* shadow)     // 阴影贴图
{

    auto& registry = Internal::GetSceneRegistry(*this);
    if (auto it = registry.find(Current); it != registry.end()) {
        it->second.instance.get().Render(window, camera, light, shadow);
    }
}
SceneMng::~SceneMng() {
}

}   // namespace CubeDemo::Scenes

