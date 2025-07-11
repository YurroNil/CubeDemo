// include/ui/main_menu/panel.h
#pragma once
#include "resources/fwd.h"
#include "ui/main_menu/base.h"

namespace CubeDemo::UI {

class MainMenuPanel : public MainMenu::MainMenuBase {
public:
    static void Init();
    static void Render();
    
    // 界面状态
    inline static bool s_isMainMenuPhase = true;
    
private:
    static void LoadScenePreviews();
    static void LoadScenelistFrom();
    static void CreatePlaceholderTexture(SceneItem& sceneItem, const string& sceneId);
};

} // namespace CubeDemo::UI
using MainMenuPanel = CubeDemo::UI::MainMenuPanel;
