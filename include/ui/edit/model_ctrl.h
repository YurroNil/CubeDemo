// include/ui/edit/model_ctrl.h
#pragma once

namespace CubeDemo {
    class Model;
}

namespace CubeDemo::UI {
class EditPanel;

class ModelCtrl {
    friend class EditPanel;

    static void Render(Camera* camera);
    static void ModelEditPanel();
    static void ModelMonitorPanel(Camera* camera);

    static void RenderModelCard(Model* model, Camera* camera);
    static void RenderModelCard(const string& modelName, float width, float height);
    static void RenderTransformControls(Model* model);
    static void RenderModelActions(Model* model, Camera* camera);

public:
    // 可用模型列表
    inline static std::vector<string> s_AvailableModels = {};

};
} // namespace CubeDemo::UI

using ModelCtrl = CubeDemo::UI::ModelCtrl;
