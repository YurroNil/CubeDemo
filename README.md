# CubeDemo


## **萌新的OpenGL试炼**

> This is a 3D game engine made by an __uneducated__ noob who just learned C++ and used the OpenGL library. It is under development currently.

## How to Play

* After Downloading make sure the file directory has the following Structure at least:  
    > ├── bin/  
    > │    └── Demo.exe  
    > │    └── <DLL Libraries Files>  
    > │  
    > └── res  
    > │    └── shaders/<Shader Files>  
    > │    └── models/<Model Files>  
    > │  
* Then just run it `bin/Demo.exe` is OK.  

## Key Description

* [**WASD**] - Move
* [**Space/Shift**] - GO Up/Down
* [**Alt**] - Call the pointer
* [**F11**] - Toggle Fullscreen Mode
* [**Mouse Wheel**] - FOV Zoom
* [**ESC**] Quit the game

## How to Compile
First you need to prepare the following some libraries into `include/` : `GLFW/`(<a href="https://github.com/glfw/glfw" target="_blank">Link</a>), `glad/`(<a href="https://glad.dav1d.de/" target="_blank">Link</a>), `glm/`(<a href="https://github.com/g-truc/glm" target="_blank">Link</a>), `json.hpp`(<a href="https://github.com/nlohmann/json" target="_blank">Link</a>), `stb_image.h`(<a href="https://github.com/nothings/stb" target="_blank">Link</a>), and Then put the `libglfw3.a`(<a href="https://github.com/glfw/glfw" target="_blank">Link</a>) into the `lib/`.  
Finally, in the root directory of the workspace Run command (Here take GCC as an example) :
```
g++.exe -fexec-charset=utf-8 -g src/*.cpp src/core/*.cpp src/rendering/*.cpp -o ./bin/Demo.exe -I"./include" -L"./lib" -lglfw3 -lopengl32 -lgdi32
```
to compile successfully.


## Changelog
* [**1.0.2 ~ 1.0.3**]
     - (2025.3.19) 完成了json导入系统(modelLoader.cpp). 添加了texture.cpp与camera.cpp.
     
* [**1.0 ~ 1.0.1**] 
     - (2025.3.17) 花了一个凌晨完成了v1.0. 热修了一波依赖库丢失的问题.
