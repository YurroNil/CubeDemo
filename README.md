# CubeDemo


## **萌新的OpenGL试炼**

> 一个职中学历的没文化菜鸟刚学会C++编程后，使用OpenGL做的小游戏.

## How to Play

* 下载好确保文件目录至少为以下结构所示：
    >* ├── bin/
    >* │    └── Demo.exe
    >* │    └── libgcc_s_seh-1.dll (程序依赖的动态库文件)
    >* │
    >* └── res
    >* │    └── shaders/<着色器文件>
    >* │    └── models/<模型文件>
    >* │
* 确认后运行`bin/Demo.exe`即可.*

## Key Description

* [**WASD**] - 移动(*会跟随视角朝向来改变Y轴)
* [**Space/Shift**] - 上升/下降
* [**Alt**] - 呼出鼠标
* [**F11**] - 全屏
* [**鼠标滚轮**] - 视角缩放
* [**ESC**] 退出游戏


## Changelog
* [**1.0.2 ~ 1.0.3**]
     - *(2025.3.19) 完成了json导入系统(modelLoader.cpp). 添加了texture.cpp与camera.cpp*
     
* [**1.0 ~ 1.0.1**] 
     - *(2025.3.17) 花了一个凌晨完成了v1.0. 热修了一波依赖库丢失的问题.*
