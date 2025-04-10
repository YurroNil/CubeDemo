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
│     │    └── modelLoader, modelMng, textureMng
│     │
│     ├── ui/ (UI模块)
│     │    └── systemMonitor, uiMng
│     │
│     ├── utils/ (工具模块)
│     │    └── root, streams (root用于包含glad、glfw、glm、string、vector、functional等高频用到的库文件; streams用于包含常用的流文件), stringConvertor
│     │
│     └── init, loop, cleanup
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
│     │     └── fragment/post/ (后处理部分, 暂无内容)
│     │     └── vertex/core/ (存放顶点着色器的glsl源码文件)
│     │     └── vertex/post/ (后处理部分, 暂无内容)
│     │
│     ├── fonts/ (字体文件)
│     │     └── (ttf文件)
│     │
│     ├── models/ (渲染模型的元数据)
│     │     └── primitives/cube.json
│     │
│     ├── textures/ (纹理文件)
│     │     └── terrain (地形纹理, 暂无内容)
│     │     └── ui (界面纹理, 暂无内容)
│     │
│     └── sounds/ (音频文件)
│           └── (储存ogg音频文件, 暂无内容)
