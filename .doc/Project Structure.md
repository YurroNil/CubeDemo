文件目录结构展示：

├── include/    (以下全部文件后缀都为.h文件)
│     │
│     ├── 3rd_party/  
│     │    └── (第三方依赖库所需的头文件)
│     │
│     ├── core/ (核心模块)
│     │     └── camera, inputHandler, keyMapper, timeMng, windowMng
│     │
│     ├── graphics/ (图形渲染模块)
│     │    └── mainRenderer, mesh, shader, shaderLoader
│     │
│     ├── resources/ (资源模块)
│     │    └── model, texture
│     │
│     ├── ui/ (UI模块)
│     │    └── systemMonitor, uiMng, debugPanel
│     │
│     ├── utils/ (工具模块)
│     │    └── streams, stringConvertor, glfwKits, glmKits, imguiKits
│     │
│     └── init, loop, cleanup, mainProgramInc
│
├── src/    (以下全部文件后缀都为.cpp文件)
│     │
│     ├── ... (目录与include保持镜像结构)
│     │
│     └── main.cpp (唯一与include不同的是src有一个main.cpp的主程序文件)
│
├── res/
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
│     ├── models/ (模型文件)
│     │     └── (obj/fbx 文件)
│     │



└── res/
      └── models/
            └── scene/
                    └── scene.obj, scene.mtl
                    └── textures/
                            └── 各种各样的纹理文件，共计405个项目，占用空间 1.56GB