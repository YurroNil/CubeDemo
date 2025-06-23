// include/scenes/default.h
#pragma once
#include "scenes/base.h"

namespace CubeDemo::Scenes {
class DefaultScene : public SceneBase {
public:
    DefaultScene();
    ~DefaultScene();

    // 覆写
    void Init() override;
    void Cleanup() override;

    // 主渲染阶段
    void Render(GLFWwindow* window, Camera* camera, ShadowMap* shadow_map) override;

private:
    DL* m_DirLight = nullptr; SkL* m_SkyLight = nullptr;
};
}   // namespace CubeDemo::Scenes
