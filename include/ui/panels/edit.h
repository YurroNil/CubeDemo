// include/ui/panels/edit.h
#pragma once
#include "managers/sceneMng.h"

namespace CubeDemo {
    class Camera;
    class Model;
}

namespace CubeDemo::UI {

class EditPanel {
// private
    // 样式备份
    inline static ImGuiStyle m_OriginalStyle = ImGuiStyle();
    inline static SceneID m_LastSceneID = SceneID::EMPTY;
    
    // 私有方法
    static void ApplyModernDarkTheme();
    static void RenderStatusBar();

public:
    // 初始化与渲染
    static void Init();
    static void Render(Camera* camera);
};
} // namespace CubeDemo::UI