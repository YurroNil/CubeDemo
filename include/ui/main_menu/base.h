// include/ui/main_menu/base.h
#pragma once
#include "scenes/base.h"

namespace CubeDemo::UI::MainMenu {

class MainMenuBase {
protected:
    // 使用与现有代码兼容的SceneItem定义
    struct SceneItem {
        string name, description, path, author, icon, previewImage;
        std::shared_ptr<Texture> preview;
    };
    
    inline static std::vector<SceneItem> m_sceneList = {};
    inline static int m_selectedScene = -1;
    inline static string m_greeting = "";
    
    // 调整卡片尺寸和间距
    static constexpr float CARD_WIDTH = 200.0f;
    static constexpr float CARD_HEIGHT = 160.0f;
    static constexpr float CARD_SPACING_X = 20.0f;
    static constexpr float CARD_SPACING_Y = 5.0f;

    // 窗口尺寸和间距
    static constexpr float SELECTION_WIDTH_RATIO = 0.25f;
    static constexpr float PREVIEW_WIDTH_RATIO = 0.5f;
    static constexpr float WINDOW_SPACING = 30.0f;
};
}
using MainMenuBase = CubeDemo::UI::MainMenu::MainMenuBase;
