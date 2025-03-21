文件目录结构展示：

├── include/
│     ├── core/
│     │     └── camera.h
│     │     └── inputHandler.h
│     │     └── windowManager.h
│     │     └── renderer.h
│     ├── rendering/
│     │    └── mesh.h
│     │    └── shader.h
│     │    └── shaderLoader.h
│     │    └── modelLoader.h
│     │    └── textRenderer.h
│     └── ui/
│          └── uiManager.h
│       
├── src/
│     ├── core/ (核心代码)
│     │     └── inputHandler.cpp
│     │     └── windowManager.cpp
│     │     └── camera.cpp
│     ├── rendering/ (图形渲染相关)
│     │     └── mesh.cpp
│     │     └── shader.cpp
│     │     └── modelLoader.cpp
│     │     └── textRenderer.cpp
│     │     └── renderer.cpp
│     ├── ui/ (界面相关)
│     │    └── uiManager.cpp
│     ├── glad.cpp
│     └── main.cpp(主程序)
│
├── res/
│     ├── animations/ (动画文件)
│     │     └── (json文件)
│     ├── shaders/ (着色器源码文件)
│     │     └── (vsh和fsh文件)
│     ├── fonts/ (字体文件)
│     │     └── (ttf文件)
│     ├── models/ (渲染模型的元数据)
│     │     └── cube.json
│     ├── textures/ (纹理文件)
│     │     └── terrain (地形纹理, 暂未使用)
│     │     └── ui (界面纹理, 暂未使用)
│     └── sounds/ (声音文件)
│           └── (暂无内容)
