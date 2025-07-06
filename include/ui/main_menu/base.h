// include/ui/main_menu/base.h
#pragma once
namespace CubeDemo::UI::MainMenu {

class MainMenuBase {
protected:
    struct SceneInfo {
        string name, description, path;
        std::shared_ptr<Texture> preview;
    };
    
    inline static std::vector<SceneInfo> m_sceneList = {};
    inline static int m_selectedScene = -1;
    inline static string m_greeting = "";
    
    // 调整卡片尺寸和间距
    static constexpr float CARD_WIDTH = 200.0f;    // 增加卡片宽度
    static constexpr float CARD_HEIGHT = 160.0f;   // 增加卡片高度
    static constexpr float CARD_SPACING_X = 20.0f; // 水平间距
    static constexpr float CARD_SPACING_Y = 5.0f; // 垂直间距

    // 窗口尺寸和间距
    static constexpr float SELECTION_WIDTH_RATIO = 0.25f;  // 选择场景窗口宽度比例
    static constexpr float PREVIEW_WIDTH_RATIO = 0.5f;     // 预览窗口宽度比例
    static constexpr float WINDOW_SPACING = 30.0f;         // 两个窗口之间的间距
};
}
using MainMenuBase = CubeDemo::UI::MainMenu::MainMenuBase;
