// include/ui/fwd.h
#pragma once

// 专用于向前声明的头文件

namespace CubeDemo::UI {
    // 普通的小型面板
    class PanelBase; class ControlPanel; class DebugPanel;

    // ↓以下均是大型面板↓

    // 预设库面板及其子功能类
    class PresetlibPanel; class PresetlistArea; class ParalistArea;

    // 暂停面板
    class PausePanel;
    // 设置面板及其子功能类
    class AboutSection; class AudioSettings; class VideoSettings; class CtrlSettings; class GameSettings; class SettingPanel;

    // 编辑模式面板及子功能类
    class EditPanel; class ScenePanel; class PresetTable; class ModelTablePanel;
}
// 乱七八糟的别名
using CMTP = CubeDemo::UI::ModelTablePanel;
