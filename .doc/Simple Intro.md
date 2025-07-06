# （一）项目介绍
项目名：Cube Demo
程序语言：C/C++ (标准库版本: C++23)
使用的图形库：OpenGL
开发依赖库：OpenGL三件套(GLFW + Glad + GLM), JSON, DearImGui, Assimp, STB_image, FontAwesome
简介：一个3D图形渲染的游戏引擎项目，使用OpenGL图形库以探索游戏开发和计算机图形学的知识，以及光线追踪的实现。
基础操作说明：WASD移动, Space/Shift上升/下降, 鼠标滚轮调整FOV, 按住ALT呼出鼠标, F11切换全屏模式
面板说明：F3调试面板，E编辑模式面板，C预设库面板，ESC暂停菜单与设置面板

项目链接：https://github.com/YurroNil/CubeDemo

# （二）文件目录结构展示

├── src/ (源文件. 以下全部文件名后缀都为cpp)
│   ├── main (主程序入口)
│   ├── core/ (核心模块)
│   │   ├── camera
│   │   ├── inputs
│   │   ├── monitor
│   │   ├── time
│   │   └── window
│   ├── loaders/ (加载器模块)
│   │   ├── async_tex
│   │   ├── font
│   │   ├── image
│   │   ├── material
│   │   ├── model
│   │   ├── model_initer
│   │   ├── progress_tracker
│   │   ├── resource
│   │   └── texture
│   ├── resources/ (资源模块)
│   │   ├── model
│   │   ├── place_holder
│   │   └── texture
│   ├── graphics/ (图形模块)
│   │   ├── bound_sphere
│   │   ├── mesh
│   │   ├── renderer
│   │   └── shader
│   ├── threads/ (线程模块)
│   │   ├── diagnostic
│   │   └── task_queue
│   ├── utils/ (工具模块)
│   │   ├── json_config
│   │   ├── string_conv
│   │   └── UTF8_to_unicode
│   ├── prefabs/ (预制体模块)
│   │   ├── shadow_map
│   │   └── lights/ (光源)
│   │       ├── data
│   │       └── volum_beam
│   ├── ui/ (UI模块)
│   │   ├── panels/
│   │   │   ├── control
│   │   │   ├── debug
│   │   │   └── pause
│   │   ├── main_menu/ (主菜单)
│   │   │   ├── bottombar
│   │   │   ├── menubar
│   │   │   ├── panel
│   │   │   ├── preview_panel
│   │   │   ├── scene_selection
│   │   │   └── title_section
│   │   ├── settings/ (设置)
│   │   │   ├── about_section
│   │   │   ├── audio
│   │   │   ├── control
│   │   │   ├── game
│   │   │   ├── panel
│   │   │   └── video
│   │   ├── presetlib/ (预设库)
│   │   │   ├── panel
│   │   │   ├── paralist_area
│   │   │   └── presetlist_area
│   │   ├── edit/ (编辑模式)
│   │   │   ├── model_table
│   │   │   ├── panel
│   │   │   ├── preset_table
│   │   │   └── scene
│   │   └── screens/
│   │       └── loading
│   ├── main/ (主程序阶段模块)
│   │   ├── cleanup
│   │   ├── handles
│   │   ├── init
│   │   ├── loop
│   │   └── rendering
│   ├── scenes/ (场景模块)
│   │   ├── dynamic_scene
│   │   └── scene_info
│   └── managers/ (管理器模块)
│       ├── ui/
│       │   └── mng
│       ├── light/
│       │   ├── creater
│       │   ├── json_mapper
│       │   ├── utils
│       │   └── mng
│       ├── model/
│       │   ├── cleanner
│       │   ├── getter
│       │   └── mng
│       └── scene/
│           └── mng
│
├── include/ (头文件. 格式为.h后缀)
│   ├── 与src保持镜像结构. 与之不同的是可能会多出一些头文件专属文件(见下面).
│   ├── 向前声明文件(格式为<源文件名>_fwd.h)
│   ├── 头文件整合文件(格式为<源文件名>_inc.h)
│   ├── 内联文件(格式为<源文件名>.inl)
│   ├── 基类文件(格式为<源文件名>_base.h)
│   ├── 宏定义文件(格式为<源文件名>_defines.h)
│   └── pch.h (预编译头文件. 包含了常用的C++标准库、第三方库, 项目中部分高频+大体积+不常修改的头文件)
│
└── resources/ (资源文件)
    ├── animations/ (动画文件)
    │   └── (动画元数据文件, 暂无内容)
    ├── shaders/ (glsl格式的着色器源码文件)
    │   ├── core/ (核心着色器)
    │   │   ├── fragment/ (片段着色器)
    │   │   ├── vertex/ (顶点着色器)
    │   │   ├── geometry/ (几何着色器)
    │   │   └── compute/ (计算着色器)
    │   └── post/ (后处理着色器)
    │       └── 文件夹结构与core相同
    ├── fonts/ (字体文件)
    │   └── (ttf文件)
    ├── models/ (模型文件，如obj, fbx文件)
    ├── images/ (图像文件)
    ├── 纹理文件/ (图像文件)
    ├── sounds/ (音频文件)
    ├── images/ (图像文件)
    └── scenes/ (场景文件)
        └── (场景名)
            ├── scene_info.json (场景配置文件)
            └── prefabs/
                └── camera.json (相机配置文件)
                └── model.json (模型配置文件)
                └── light.json (光源配置文件)
