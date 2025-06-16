// include/managers/sceneMng.h
#pragma once
#include "managers/fwd.h"
#include "managers/scene/getter.h"
#include "managers/lightMng.h"

using VolumBeam = CubeDemo::Prefabs::VolumBeam;

namespace CubeDemo::Managers {

class SceneMng : public SceneGetter {
    friend class SceneGetter;
public:
    SceneMng();
    ~SceneMng();

    enum class SceneID { DEFAULT, NIGHT, EMPTY } Current;

    // 核心方法改为模板调用
    void Init();
    void CleanAllScenes();
    void SwitchTo(SceneID target);
    // 场景渲染
    void Rendering(
        SceneID current,       // 场景ID
        GLFWwindow* window,    // 窗口
        Camera* camera,        // 摄像机
        ShadowMap* shadow      // 阴影贴图
    );

    // 场景实例
    Scenes::DefaultScene Default; 
    Scenes::NightScene Night;
    SceneGetter GetCurrentScene = this;

    // 场景管理器实例创建/删除
    static SceneMng* CreateInst();
    static void RemoveSceneInst(SceneMng* ptr);

    string PrintCurrent(SceneID& inst);
    
private:
    // 统计场景管理器的数量. 不允许存在多个场景管理器
    inline static unsigned int s_InstCount = 0;
};

using SceneID = SceneMng::SceneID;
}   // namespace CubeDemo::Scenes

// 模板实现
#include "managers/scene/utils.inl"

namespace CubeDemo {
    using SceneID = Managers::SceneMng::SceneID;
}
