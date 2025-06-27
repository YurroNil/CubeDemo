// include/ui/panels/pause.h
#pragma once

namespace CubeDemo::UI {

class PausePanel {
// private
    // 菜单尺寸
    inline static ImVec2 m_MenuSize = ImVec2(400, 500);
    
    // 私有渲染函数
    static void RenderBackground(ImVec2 p_min, ImVec2 p_max, float corner_rounding);
    static void RenderTitleArea();
    static void RenderSearchBar();
    static void RenderContentArea(const ImVec2& menuSize);
    static void RenderBottomButtons(GLFWwindow* window, float buttonAreaTopY);
    static void RenderCopyright(float buttonAreaTopY);

public:
    // 初始化与渲染
    static void Init();
    static void Render(GLFWwindow* window);
};

} // namespace CubeDemo::UI