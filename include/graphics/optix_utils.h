// src/graphics/optix_utils.h
#pragma once
// 启用optix 9.0的"无静态链接库 仅头文件"模式
#define OPTIX_HEADERS_ONLY 1

#include <optix.h>
#include <optix_stubs.h>
#include "graphics/cuda_utils.h"

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX error at " << __FILE__ << ":" << __LINE__ << " - " << optixGetErrorString(res) << " (" #call ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
}

struct GASBuffer {
    CUdeviceptr d_buffer = 0;
    OptixTraversableHandle handle = 0;
};

struct BuildInput {
    OptixBuildInput input = {};
    std::vector<OptixInstance> instances;
};

inline GASBuffer build_gas(
    OptixDeviceContext context,
    OptixBuildInput& build_input,
    OptixAccelBuildOptions accel_options
) {
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &build_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));

    GASBuffer result;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&result.d_buffer),
        gas_buffer_sizes.outputSizeInBytes
    ));

    // Temporary build buffer
    CUdeviceptr d_temp_buffer = 0;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        gas_buffer_sizes.tempSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &build_input,
        1, // Number of build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        result.d_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &result.handle,
        nullptr, // emitted properties
        0 // num emitted properties
    ));

    // Free temporary buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    
    return result;
}

inline void free_gas(GASBuffer& gas) {
    if (gas.d_buffer) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(gas.d_buffer)));
        gas.d_buffer = 0;
        gas.handle = 0;
    }
}
