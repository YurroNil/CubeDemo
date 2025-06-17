// include/ui/panels/edit.h
#pragma once
#include "core/inputs.h"
#include "utils/string_conv.h"
#include "managers/sceneMng.h"

namespace CubeDemo::UI {

class EditPanel {
public:
    static void Render();
    
private:
    static void SceneMngment();
    static void ModelMonitoring();
    static void ModelEditing();
    static void ModelEditPanel();
    static void ModelMonitorPanel();
    static void ScenePanel();
    // 模型数据结构
    struct ModelInfo {
        string id, name, type;
        vec3 position;
        float rotation;
        vec3 scale;
    };
    
    static std::vector<ModelInfo> s_ModelInfos;
    static std::vector<string> s_AvailableModels;
};
} // namespace CubeDemo::UI
