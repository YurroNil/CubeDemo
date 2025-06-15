文件目录结构展示：

├── include/    (以下全部文件后缀都为.h文件)
│     │
│     ├── core/ (核心模块)
│     │    └── camera, inputs, time, window, monitor
│     │
│     ├── graphics/ (图形渲染模块)
│     │    └── base, bound_sphere, renderer, mesh, shader
│     │
│     ├── resources/ (资源模块)
│     │    └── model, textureBase, texture, placeHolder
│     │
│     ├── threads/ (线程模块)
│     │    └── diagnostic, task_queue
│     │
│     ├── loaders/ (加载器模块)
│     │    └── base, material, model, model_initer, resource, texture, image, async_tex, fonts
│     │
│     ├── ui/ (UI模块)
│     │    └── panels/
│     │         └── pause, debug, control
│     │
│     ├── scenes/ (场景布置模块)
│     │    └── base, default, night
│     │
│     ├── prefabs/ (预制体模块，如光源、雨雪、烟雾、粒子效果等计算几何体)
│     │    ├── lights/
│     │    │    └── base, volum_beam
│     │    └── shadow_map
│     │
│     ├── managers/ (管理器模块)
│     │    ├── light/
│     │    │    └── cleanner, creater, getter
│     │    ├── model/
│     │    │    └── cleanner, creater, getter
│     │    ├── scene/
│     │    │    └── getter
│     │    └── lightMng, modelMng, sceneMng, uiMng
│     │
│     ├── utils/ (工具模块)
│     │    └── string_conv, json_config, defines
│     │
│     ├── main/ (main主程序模块)
│     │    └── init, loop, cleanup, rendering, handles
│     └── main
│
├── src/    (以下全部文件后缀都为.cpp文件)
│     │
│     ├── ... (目录与include保持镜像结构)
│     └── main.cpp
│          1. `src/`与`include/`不同的是: src独有main.cpp
│          2. 没有任何头文件独有的镜像源文件, 如: 带`base`, `fwd(向前声明文件)`, `defines(宏命令文件)`, `pch(预编译头文件)`等字样。
│
├── resources/
│     ├── animations/ (动画文件)
│     │     └── (动画元数据文件, 暂无内容)
│     │
│     ├── shaders/ (着色器源码文件)
│     │     └── fragment/core/ (存放片段着色器的glsl源码文件)
│     │     └── vertex/core/ (存放顶点着色器的glsl源码文件)
│     │
│     ├── fonts/ (字体文件)
│     │     └── (ttf文件)
│     │
└─────└── models/ (模型文件，如obj, fbx文件)
            └── sample/
                    └── sample.obj, sample.mtl
                    └── textures/
                            └── 各种各样的纹理文件
