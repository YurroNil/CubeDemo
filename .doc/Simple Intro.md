# （一）项目介绍
项目名：Cube Demo
程序语言：C/C++ (标准库版本: C++23)
使用的图形库：OpenGL 4.6
开发依赖库：OpenGL三件套(GLFW + Glad + GLM), JSON, DearImGui, Assimp, STB_image, CUDA_ToolKits, OptiX 9.0
简介：一个3D图形渲染的游戏引擎项目，使用OpenGL图形库以探索游戏开发和计算机图形学的知识，以及光线追踪的实现。
基础操作说明：WASD移动, Space/Shift上升/下降, 鼠标滚轮调整FOV, 按住ALT呼出鼠标, F11切换全屏模式
面板说明：F3调试面板，E编辑模式面板，C预设库面板，ESC暂停菜单与设置面板

项目链接：https://github.com/YurroNil/CubeDemo

# （二）CubeDemo进度
CubeDemo目前处于开发阶段，功能尚不完整，但部分基本功能已实现。

## 已实现内容
1. 常用文件加载: 模型文件(如obj, fbx, gltf等, 使用Assimp库), 图像文件(如png, jpg, bmp等, 使用STB_image库), 字体文件(如ttf, otf等, 使用FreeType库).
2. 配置文件加载: 在resources中定义了各种配置文件的结构(json格式), 并实现了配置文件的加载与保存. 如: 场景信息、场景对象与模型属性、摄像机属性、光源属性、字体范围的配置文件. resources/scenes下可创建自定义文件夹并且放入结构正确的scene_info.json配置文件后, 即可在主菜单的场景列表中选择并加载.
3. UI渲染: 实现了基本的UI渲染(基于DearImGui库), 包括主界面菜单、编辑面板、暂停与设置面板、预设库面板、调试面板、控制面板. 并实现了部分UI的交互逻辑, 如: 场景、预制体和模型的创建、删除、编辑等.
4. 图形渲染: 实现了基本的光照模型(基于Phong模型), 点光, 聚光, 环境光, 灯光衰减, 反射等.

## TODO
1. 实现全局光照(屏幕光遮蔽与光线追踪): 包括阴影, 全局光照衰减, 全局光照遮挡, 全局光照反射等.
2. 骨骼与动画系统: 实现了骨骼系统(基于Assimp库), 包括骨骼重定向、绑定、关键帧、线性/非线性动画等.
3. 物理引擎: 实现基本的物理引擎(基于Bullet库), 包括刚体、碰撞体、触发体、摩擦力、重力、弹簧、滑动摩擦力、碰撞反馈、碰撞检测等.
4. 性能优化
    - 完善LOD系统;
    - 异步资源加载与多线程管理;
    - 使用计算着色器优化性能;
    - 实现遮挡剔除算法;
    - 实现实例化渲染系统.

# （三）文件目录结构展示

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
│   │   ├── texture
│   │   ├── material
│   │   └── mesh
│   ├── graphics/ (图形模块)
│   │   ├── bound_sphere
│   │   ├── ray_tracing
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
│   │   └── lights
│   │   └── volum_beam
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
│   │   ├── init
│   │   ├── loop
│   ├── scenes/ (场景模块)
│   │   ├── dynamic_scene
│   │   └── scene_info
│   └── managers/ (管理器模块)
│       ├── uiMng
│       ├── light_utils
│       ├── json_mapper
│       ├── lightMng
│       ├── modelMng
│       ├── model_getter
│       └── sceneMng
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
    ├── textures/ (纹理文件)
    ├── sounds/ (音频文件)
    ├── images/ (图像文件)
    └── scenes/ (场景文件)
        └── (场景名)
            ├── scene_info.json (场景配置文件)
            └── prefabs/
                └── camera.json (相机配置文件)
                └── model.json (模型配置文件)
                └── light.json (光源配置文件)
