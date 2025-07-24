// include/pch.h
#pragma once

/*
    该文件是预编译头文件(Precompiled Header File)，包含了第三方库、标准库以及项目中常包含的头文件，以便提高编译速度.
    注意：必须要包含更新频率较低的头文件，否则每次更新这个文件，都要导致预编译头文件(pch.h.gch)重新生成. 请谨慎更新！！
*/

/* ---------C++标准库--------- */
    #include <vector>
    #include <atomic>
    #include <memory>
    #include <functional>
    #include <tuple>
    #include <future>
    #include <mutex>
    #include <thread>
    #include <unordered_map>
    #include <map>
    #include <future>
    #include <array>
    #include <deque>
    #include <queue>
    #include <unordered_set>
    #include <windows.h>
    #include <algorithm>
    #include "kits/strings.h"
    #include "kits/streams.h"
    #include "kits/file_system.h"
    #include "kits/times.h"

/* ---------第三方库--------- */
    #include "kits/assimp.h"
    #include "kits/glfw.h"
    #include "kits/glm.h"
    #include "kits/imgui.h"
    #include "kits/json.h"

/* ---------项目头文件--------- */
    // 核心模块
    #include "core/camera.h"
    #include "core/window.h"
    #include "core/inputs.h"
    #include "core/time.h"
    // 线程模块
    #include "threads/diagnostic.h"
    // 图形模块
    #include "graphics/renderer.h"
    #include "resources/mesh.h"
    #include "graphics/shader.h"
    // 工具模块
    #include "utils/json_config.h"
    #include "utils/string_conv.h"
    // 预制体模块
    #include "prefabs/lights.h"

