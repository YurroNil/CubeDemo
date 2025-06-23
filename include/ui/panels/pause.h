// include/ui/panels/pause.h
#pragma once
namespace CubeDemo::UI {
class PausePanel {
// private
    static void SetMenuContent(GLFWwindow* window);

public:
    static void Render(GLFWwindow* window);
};
}   // namespace CubeDemo::UI
