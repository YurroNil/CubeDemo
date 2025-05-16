// include/scenes/mainScene.h

#pragma once
#include "scenes/sceneBase.h"
#include "graphics/shadowMap.h"

namespace CubeDemo::Scenes {
class MainScene : public SceneBase {
public:
    // 覆写
    void Init() override;
    void Render(GLFWwindow* window, Camera* camera) override {};
    void Cleanup() override;
    ~MainScene();

    // 普通成员
    bool isInited = false, isCleanup = false;

    // 阴影渲染阶段
    void RenderShadow(Camera* camera);

    // 主渲染阶段
    void RenderMainPass(GLFWwindow* window, Camera* camera);

    // Getters
    GSM* GetShadowMap() const;
    Shader* GetShadowShader() const;
    DL* GetDirLight() const;

private:
    // 场景私有资源
    GSM* m_ShadowMap = nullptr;
    Shader* m_ShadowShader = nullptr;
    DL* m_DirLight = nullptr;
};
}
