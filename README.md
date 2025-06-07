# CubeDemo

## **萌新的OpenGL试炼**

> This is a 3D Game Engine made by an __uneducated__ noob who just learned C++ and used the OpenGL Framework. It is under development currently.

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
* [**1.0.0**]
     - (2025.03.17)
          -- 完成了基本的WASD, 全屏与立方体渲染.
     - (2025.03.23)
          -- 增加了光照效果.
     - (2025.05.15)  
          -- 使用Assimp库完成了加载obj与fbx等格式的自定义模型;  
          -- 完成了部分的异步加载系统(未启用);  
          -- 可以修改resources文件夹内config.json和custom_chars.json的数据, 以热更新加载程序中的模型与字符.  
