文件目录结构展示：

├── include/    (以下全部文件后缀都为.h文件)
│     │
│     ├── core/ (核心模块)
│     │     └── camera, inputs, time, window, monitor
│     │
│     ├── graphics/ (图形渲染模块)
│     │    └── boundingSphere, renderer, mesh, shader, lod
│     │
│     ├── resources/ (资源模块)
│     │    └── model, texture, placeHolder
│     │
│     ├── threads/ (线程模块)
│     │    └── diagnostic, taskQueue, loaders
│     │
│     ├── loaders/ (加载器模块)
│     │         └── material, model, resource, texture, image, asyncTex
│     │
│     ├── ui/ (UI模块)
│     │    └── monitor, uiMng
│     │
│     ├── utils/ (工具模块)
│     │    └── stringConvertor, utf8_to_unicode, msaConv, jsonConfig, defines
│     │
│     ├── main/ (main主程序工具链模块)
│     │         └── includes
│     └── init, loop, cleanup
│
├── src/    (以下全部文件后缀都为.cpp文件)
│     │
│     ├── ... (目录与include保持镜像结构)
│     │
│     └── main.cpp (唯一与include不同的是src有一个main.cpp的主程序文件)
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
