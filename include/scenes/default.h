// include/scenes/default.h
#pragma once
#include "scenes/base.h"

namespace CubeDemo::Scenes {
class DefaultScene : public SceneBase {
public:
    DefaultScene() = default;
    ~DefaultScene();

    // 覆写
    void Init(SceneMng* scene_inst, Light& light) override;
    void Cleanup(Light& light) override;

    // 主渲染阶段
    void Render(
        GLFWwindow* window,
        Camera* camera,
        const Light& light,
        ShadowMap* shadow_map
    ) override;
};
}   // namespace CubeDemo::Scenes
