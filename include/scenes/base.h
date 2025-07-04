// include/scenes/base.h
#pragma once
#include "scenes/fwd.h"
#include "prefabs/lights/fwd.h"
#include "managers/fwd.h"

// 向前声明
namespace CubeDemo {
    class Camera; class Shader;
}
// 乱七八糟的别名
using Camera = CubeDemo::Camera;
using Shader = CubeDemo::Shader;

namespace CubeDemo::Scenes {

// 场景基类
class SceneBase {
    friend SceneMng;
public:
    virtual string GetName() const { return name; }
    virtual string GetID() const { return id; }
    
    virtual ~SceneBase() = default;  // 虚析构函数

    // 场景初始化（加载资源）
    virtual void Init() = 0;

    // 场景渲染
    virtual void Render(GLFWwindow*, Camera*, ShadowMap*) = 0;

    // 场景逻辑更新
    virtual void Update() {}
    
    // 场景资源清理
    virtual void Cleanup() = 0;

     // 状态查询接口
    virtual bool IsInited() const { return m_isInited; }
    virtual bool IsCleanup() const { return m_isCleanup; }
 
protected:
    bool m_isInited = false, m_isCleanup = false;
    string name = "unnamed", id  = "none";

};
}   // namespace
