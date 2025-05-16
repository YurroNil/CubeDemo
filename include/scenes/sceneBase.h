// include/scenes/scene.h

#pragma once
#include "core/camera.h"
#include "graphics/shadowMap.h"
#include "resources/model.h"
#include "core/window.h"
#include "graphics/shadowMap.h"

using GSM = CubeDemo::Graphics::ShadowMap;
using DL = CubeDemo::Graphics::DirLight;

namespace CubeDemo::Scenes {

using Camera = CubeDemo::Camera;
using Shader = CubeDemo::Shader;

// 场景基类
class SceneBase {
public:
    // 场景初始化（加载资源）
    virtual void Init() {}
    virtual void Render(GLFWwindow* window, Camera* camera) = 0;
    // 场景逻辑更新
    virtual void Update() {}
    // 场景资源清理
    virtual void Cleanup() = 0;
    
    struct Creater {
        // 阴影着色器工厂方法
        static Shader* ShadowShader();
        // 阴影贴图工厂方法
        static GSM* ShadowMap();
        // 静态顶点光源创建
        static DL* DirLight();

    } Create;
};

}