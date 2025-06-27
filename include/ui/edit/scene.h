// include/ui/edit/scene.h
#pragma once

namespace CubeDemo::UI {
class EditPanel;

class ScenePanel {
    friend class EditPanel;
    static void Render();
    static void RenderScenePreview();
};
} // namespace CubeDemo::UI

using ScenePanel = CubeDemo::UI::ScenePanel;
