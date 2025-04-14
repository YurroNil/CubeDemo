# CubeDemo


## **萌新的OpenGL试炼**

> This is a 3D game engine made by an __uneducated__ noob who just learned C++ and used the OpenGL library. It is under development currently.

## How to Play

* After Downloading make sure the file directory has the following Structure at least:  
    > ├── bin/  
    > │    └── Demo.exe  
    > │    └── (DLL Libraries Files)  
    > │  
    > ├── resources/  
    >     └── (Animations, Fonts, Shaders, Models, Textures... etc)  
 
* Then just run it `Demo.exe` is OK.  

## Key Description

* [**WASD**] - Move
* [**Space/Shift**] - Go Up/Down
* [**Alt (Holding)**] - Call the Cursor
* [**F11**] - Toggle Fullscreen Mode
* [**Mouse Wheel**] - FOV Zoom
* [**ESC**] - Open/Close the Menu
* [**F3**] - Open/Close the Debuginfo

## How to Compile
First you need to prepare the following some libraries into `include/` : `GLFW` (<a href="https://github.com/glfw/glfw" target="_blank">Link</a>) , `glad` (<a href="https://glad.dav1d.de/" target="_blank">Link</a>) , `glm` (<a href="https://github.com/g-truc/glm" target="_blank">Link</a>) , `json` (<a href="https://github.com/nlohmann/json" target="_blank">Link</a>) , `stb` (<a href="https://github.com/nothings/stb" target="_blank">Link</a>) and `assimp` (<a href="https://github.com/assimp/assimp" target="_blank">Link</a>).  
Finally, in the root directory of the workspace Run command (Here take GCC as an example) :
```
g++.exe -g\
     "../src/core"\
     "../src/graphics"\
     "../src/resources"\
     "../src/ui"\
     "../src/utils"\
-o ./bin/Demo.exe\
     -Iinclude -Llib -lglfw3 -lopengl32 -lassimp
```
to compile successfully.


## Changelog
* [**1.0.6**]
     - (2025.04.04) 增加了暂停菜单; 代码更加模块化; 增加了对象注册模块; 增加了对象变换模块(如移动、缩放、旋转等).

* [**1.0.5**]
     - (2025.03.23) 增加了光照效果.

* [**1.0.4**]
     - (2025.03.21) 为了拓展性使项目模块化. 完成了字体渲染类. 打开F3能显示调试面板.

* [**1.0.2 ~ 1.0.3**]
     - (2025.03.19) 完成了json导入系统(modelLoader.cpp). 添加了texture.cpp与camera.cpp.
     
* [**1.0 ~ 1.0.1**] 
     - (2025.03.17) 花了一个凌晨完成了v1.0. 热修了一波依赖库丢失的问题.
