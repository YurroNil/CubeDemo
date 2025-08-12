// include/kits/cuda_and_optix.h
#pragma once

// 先处理宏冲突
#ifdef __cdecl
#undef __cdecl
#endif

// 再包含CUDA头文件
#include <cuda.h>            // 添加CUDA驱动API头文件
#include <cuda_gl_interop.h> // CUDA-OpenGL互操作
#include <cuda_runtime.h>
#include <curand_kernel.h>

// 启用optix 9.0的"无静态链接库 仅头文件"模式
#define OPTIX_HEADERS_ONLY 1
#include <optix.h>
#include <optix_stubs.h>
