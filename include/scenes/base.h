// include/scenes/sceneBase.h

#pragma once
#include "core/camera.h"
#include "prefabs/shadowMap.h"
#include "resources/model.h"
#include "core/window.h"

namespace CubeDemo::Scenes {

// 乱七八糟的别名
using DL = CubeDemo::Prefabs::DirLight;
using Camera = CubeDemo::Camera;
using Shader = CubeDemo::Shader;
using Light = Prefabs::Light;
using ShadowMap = Prefabs::ShadowMap;

// 向前声明
class SceneMng; class DefaultScene; class NightScene;

// 场景基类
class SceneBase {
    friend SceneMng;
public:
    virtual ~SceneBase() = default;  // 虚析构函数

    // 场景初始化（加载资源）
    virtual void Init(SceneMng*, Light&) = 0;

    // 场景渲染
    virtual void Render(GLFWwindow*, Camera*, const Light&, ShadowMap*) = 0;

    // 场景逻辑更新
    virtual void Update() {}
    
    // 场景资源清理
    virtual void Cleanup(Light&) = 0;

     // 新增状态查询接口
    virtual bool IsInited() const { return s_isInited; }
    virtual bool IsCleanup() const { return s_isCleanup; }
 
protected:
    bool s_isInited{false};
    bool s_isCleanup{false};

};
}   // namespace CubeDemo::Scenes
