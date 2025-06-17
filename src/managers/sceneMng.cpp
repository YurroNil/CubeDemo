// src/managers/scene.cpp
#include "pch.h"
#include "managers/sceneMng.h"

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
        inst.s_isInited = true; inst.s_isCleanup = false;
    }
}
// 清理所有场景
void SceneMng::CleanAllScenes() {
    Default.Cleanup();
    Night.Cleanup();
    // 扩展时添加新场景清理
}

string SceneMng::PrintCurrent(SceneID& inst) {
    switch (inst) {
    case SceneID::DEFAULT:
        return Default.name;
    case SceneID::NIGHT:
        return Night.name;
    default: return "null";
    }
    return "?";
}

// 切换场景
void SceneMng::SwitchTo(SceneID target) {
    // 每次切换场景时，清理所有资源
    CleanAllScenes();

    std::cout << "要切换的目标场景: " << PrintCurrent(target) << "\n" << std::endl;
    Current = target;

    // 然后再重新初始化
    Init();
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
    if(s_InstCount > 0) {
        std::cerr << "[SceneMng] 场景创建失败，因为当前场景管理器数量为: " << s_InstCount << std::endl;
        return nullptr;
    }
    s_InstCount++;
    return new SceneMng();
}
// 删除场景管理器
void SceneMng::RemoveInst(SceneMng** ptr) {
    if(s_InstCount = 0) return;
    s_InstCount--;
    delete *ptr; *ptr = nullptr;
}

}   // namespace CubeDemo::Scenes
