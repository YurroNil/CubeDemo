// include/ui/edit/scene.h
#pragma once
#include "managers/scene/mng.h"

namespace CubeDemo::UI {
class EditPanel;

class ScenePanel {
    friend class EditPanel;
    static void ScenePreview();
    static void SaveCurrentScene();
    static void ClearCurrentScene();

public:
    static void Render();

};
} // namespace CubeDemo::UI

using ScenePanel = CubeDemo::UI::ScenePanel;
