// include/ui/panels/debug.h
#pragma once

namespace CubeDemo {
    class Camera;
}
namespace CubeDemo::UI {

class DebugPanel {
public:
    static void Render(Camera* camera);
};

}