// include/ui/edit/panel.h
#pragma once
#include "managers/scene.h"

namespace CubeDemo {
    class Camera;
    class Model;
}

namespace CubeDemo::UI {

class EditPanel {
// private
    // 样式备份
    inline static ImGuiStyle m_OriginalStyle = ImGuiStyle();

    static void ApplyModernDarkTheme();
    static void CtrlPanel(Camera* camera);

public:
    // 初始化与渲染
    static void Init();
    static void Render(Camera* camera);
};
} // namespace CubeDemo::UI