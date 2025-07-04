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

## Changelog
* [**1.0.7**]
     - (2025.05.15)  
          -- 使用Assimp库完成了加载obj与fbx等格式的自定义模型;  
          -- 完成了部分的异步加载系统(未启用); 完成了初步的LOD系统(未启用);  
          -- 可以修改resources文件夹内config.json和custom_chars.json的数据, 以热更新加载程序中的模型与字符.  

* [**1.0.6**]
     - (2025.04.04) 增加了暂停菜单; 代码更加模块化; 增加了对象注册模块; 增加了对象变换模块(如移动、缩放、旋转等).

* [**1.0.5**]
     - (2025.03.23) 增加了光照效果.

* [**1.0.4**]
     - (2025.03.21) 为了拓展性使项目模块化. 完成了字体渲染类. 打开F3能显示调试面板.

* [**1.0.2 ~ 1.0.3**]
     - (2025.03.19) 完成了json导入系统(modelLoader.cpp). 添加了texture.cpp与camera->cpp.
     
* [**1.0 ~ 1.0.1**] 
     - (2025.03.17) 花了一个凌晨完成了v1.0. 热修了一波依赖库丢失的问题.
