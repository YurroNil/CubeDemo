// include/ui/control_panel.h
#pragma once

namespace CubeDemo {
    class Camera;
}
namespace CubeDemo::UI {

class ControlPanel {
public:
    static void Render(Camera* camera);
};

}