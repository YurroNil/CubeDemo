// include/managers/ui_inc.h
#pragma once
/*
    uiMng因为涉及到多个模块的类，因此需要专门做个inc类整合
*/

// 管理器模块
#include "managers/ui.h"
// 界面模块
#include "ui/edit/panel.h"
#include "ui/edit/model_table.h"
#include "ui/panels/pause.h"
#include "ui/panels/control.h"
#include "ui/panels/debug.h"
#include "ui/presetlib/panel.h"
#include "ui/main_menu/panel.h"
// 加载器模块
#include "loaders/font.h"
#include "loaders/progress_tracker.h"
