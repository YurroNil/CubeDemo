// include/ui/panels/pause.h
#pragma once
namespace CubeDemo::UI {
class PausePanel {
public:
    static void Render(GLFWwindow* window);
private:
    static void SetMenuContent(GLFWwindow* window);
};
}