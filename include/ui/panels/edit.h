// include/ui/panels/edit.h
#pragma once
namespace CubeDemo::UI {

class EditPanel {
// private
    inline static SceneID m_LastSceneID = SceneID::EMPTY;
    inline static ImVec4
        m_OriginalBgColor, m_OriginalChildBg,
        m_OriginalPopupBg, m_OriginalFrameBg, m_OriginalFrameBgHovered, m_OriginalFrameBgActive;

    static void ModelEditPanel();
    static void ModelMonitorPanel(Camera* camera);
    static void ScenePanel();
    static void ModelControlPanel(Camera* camera);
    
public:
    static void Init();
    static void Render(Camera* camera);
    inline static std::vector<string> s_AvailableModels;

};
} // namespace CubeDemo::UI
