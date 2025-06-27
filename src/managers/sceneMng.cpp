// src/managers/scene.cpp
#include "pch.h"
#include "ui/panels/edit.h"
#include "ui/edit/model_ctrl.h"
#include "managers/modelMng.h"
#include "loaders/texture.h"

// 外部变量声明
namespace CubeDemo {
    extern ModelMng* MODEL_MNG;
}

namespace CubeDemo::Managers {

// 别名
namespace Internal = SceneInternal;

// 初始化
void SceneMng::Init() {

    auto& registry = Internal::GetSceneRegistry(this);
    if (auto it = registry.find(Current); it != registry.end()) {
        // 获取SceneMng的内部场景实例
        auto& inst = it->second.instance;
        // 调用初始化函数
        inst.Init();
        inst.m_isInited = true; inst.m_isCleanup = false;
    }
}
// 清理所有场景
void SceneMng::CleanAllScenes() {
    // 删除场景资源 (如光源预制体)
    Default.Cleanup();
    Night.Cleanup();

    // 删除所有模型与着色器
    MODEL_MNG->RmvAllModels();
    
    // 清除纹理缓存
    TL::ClearCache();

    // 重置UI状态
    UI::ModelCtrl::s_AvailableModels.clear();
}

// 切换场景
void SceneMng::SwitchTo(SceneID target) {
    // 清理所有场景
    CleanAllScenes();
    
    // 更新当前场景
    Current = target;
    
    // 初始化新场景
    Init();
    
    // 更新UI模型列表
    UI::ModelCtrl::s_AvailableModels = GetCurrentScene.ModelNames();
}

// 渲染
void SceneMng::Rendering(
    SceneID current,       // 场景ID
    GLFWwindow* window,    // 窗口
    Camera* camera,        // 摄像机
    ShadowMap* shadow)     // 阴影贴图
{
    auto& registry = Internal::GetSceneRegistry(this);
    if (auto it = registry.find(Current); it != registry.end()) {
        it->second.instance.Render(window, camera, shadow);
    }
}
SceneMng::SceneMng()
    : SceneGetter(this) {
    // 场景对象初始化
    Default = Scenes::DefaultScene();
    Night = Scenes::NightScene();
    
    // 强制初始化注册表
    auto& _ = Internal::GetSceneRegistry(this);
    }

SceneMng::~SceneMng() {}

// 创建场景管理器
SceneMng* SceneMng::CreateInst() {
    if(m_InstCount > 0) {
        std::cerr << "[SceneMng] 场景创建失败，因为当前场景管理器数量为: " << m_InstCount << std::endl;
        return nullptr;
    }
    m_InstCount++;
    return new SceneMng();
}
// 删除场景管理器
void SceneMng::RemoveInst(SceneMng** ptr) {
    if(m_InstCount = 0) return;
    m_InstCount--;
    delete *ptr; *ptr = nullptr;
}

}   // namespace CubeDemo::Scenes
