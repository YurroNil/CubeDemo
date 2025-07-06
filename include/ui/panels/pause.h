// include/ui/panels/pause.h
#pragma once

namespace CubeDemo::UI {
class PausePanel {
// private
    // 菜单尺寸
    inline static ImVec2 m_MenuSize = ImVec2(400, 500);
    
    // 私有渲染函数
    static void Render(ImVec2 p_min, ImVec2 p_max, float corner_rounding);
    static void TitleArea();
    static void SearchBar();
    static void ContentArea(const ImVec2& menu_size);
    static void BottomButtons(GLFWwindow* window, float button_area_topY);
    static void Copyright(float button_area_topY);

public:
    // 初始化与渲染
    static void Init();
    static void Render(GLFWwindow* window);
};

} // namespace CubeDemo::UI