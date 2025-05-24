// include/scenes/sceneMng.h
#pragma once
#include "scenes/default.h"
#include "scenes/night.h"

namespace CubeDemo::Scenes {

class SceneMng {
public:
    SceneMng() : Default(), Night() {}
    ~SceneMng();

    enum class SceneID { DEFAULT, NIGHT, EMPTY } Current;

    // 核心方法改为模板调用
    void Init(Light& light);
    void CleanAllScenes(Light& light);
    void SwitchTo(SceneID target, Light& light);

    // 场景渲染
    void Rendering(
        SceneID current,       // 场景ID
        GLFWwindow* window,    // 窗口
        Camera* camera,        // 摄像机
        const Light& light,    // 光源
        ShadowMap* shadow      // 阴影贴图
    );

    // 场景实例
    DefaultScene Default; 
    NightScene Night;

private:
    // 私有工具方法
    template <SceneID id>
    void CleanScene(Light& light);
};

using SceneID = SceneMng::SceneID;
}   // namespace CubeDemo::Scenes

// 模板实现
#include "scenes/sceneUtils.inl"
#include "scenes/sceneMng.inl"
