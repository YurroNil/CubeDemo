// include/managers/scene/mng.h
#pragma once
#include "managers/fwd.h"
#include "scenes/base.h"
#include "utils/defines.h"

namespace CubeDemo::Managers {


class SceneMng {
public:
    SceneMng() {}
    ~SceneMng();
    
    void Init(const string& sceneDir = SCENE_CONF_PATH);
    void SwitchTo(const string& sceneID);
    void Cleanup();
    Scenes::SceneBase* GetCurrentScene() const { return m_currentScene; }
    const auto& GetAllScenes() const { return m_scenes; }
    
    static SceneMng* CreateInst();
    static void RemoveInst(SceneMng** ptr);
    
private:
    void LoadSceneConfigs();
    void ParsingData(const fs::path& sceneDir);
    
    std::unordered_map<string, Scenes::SceneBase*> m_scenes;
    Scenes::SceneBase* m_currentScene = nullptr;
    string m_sceneDir;
    
    inline static unsigned int m_InstCount = 0;
};

} // namespace CubeDemo::Managers
