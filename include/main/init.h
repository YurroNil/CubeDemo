// include/main/init.h
#pragma once

namespace CubeDemo {

// 程序初始化
GLFWwindow* Init(int argc, char* argv[]);
void parsing_arguments(int argc, char* argv[]);
void init_program_core();
void init_camera();
void init_managers();
}
