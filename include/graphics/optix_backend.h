// include/graphics/optix_backend.h
#pragma once
#include <cuda_runtime.h>
#include "graphics/optix_utils.h"

namespace CubeDemo {

// 定义OptiX使用的数据结构
struct OptixTriangle {
    float3 v0, v1, v2;
    float3 n0, n1, n2;
    float2 uv0, uv1, uv2;
    int materialIndex;
    float3 emission;
};

struct OptixMaterial {
    float3 diffuse;
    float3 specular;
    float3 emission;
    float shininess;
    float opacity;
};

// 相机参数
struct CameraParams {
    float3 eye;
    float3 lookat;
    float3 up;
    float fov;
    float aspect;
};

// 着色器绑定表记录
struct RaygenRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct MissRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitgroupRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// 启动参数
struct Params {
    OptixTraversableHandle handle;
    cudaArray_t image;
    int width;
    int height;
    float3 eye;
    float3 lookat;
    float3 up;
    float fov;
    float aspect;
    OptixTriangle* triangles;
    OptixMaterial* materials;
};

class OptixBackend {
public:
    static OptixBackend& GetInstance();
    
    void Init(int width, int height);
    void Shutdown();
    
    void UploadScene(
        const std::vector<OptixTriangle>& triangles,
        const std::vector<OptixMaterial>& materials
    );
    
    void UpdateCamera(
        const vec3& position,
        const vec3& lookAt,
        const vec3& up,
        float fov,
        float aspect
    );
    
    void Render();
    
    cudaGraphicsResource_t GetOutputTextureResource() { return cuda_texture_resource_; }
    unsigned int GetOutputTexture() { return output_texture_; }

private:
    OptixBackend();
    ~OptixBackend();
    
    void CreatePipeline();
    void SetupSBT();
    void BuildAccelerationStructure();
    
    OptixDeviceContext context_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_ = {}; // 只定义一次
    
    cudaStream_t stream_ = nullptr;
    cudaGraphicsResource_t cuda_texture_resource_ = nullptr;
    OptixModule module_ = nullptr;
    unsigned int output_texture_ = 0;
    
    // 场景数据
    CUdeviceptr d_triangles = 0;
    CUdeviceptr d_materials = 0;
    GASBuffer gas_ = {}; // 使用GASBuffer类型
    
    int width_ = 0;
    int height_ = 0;

    // 内部状态
    CameraParams camera_params_;
    size_t triangle_count_ = 0;
    
    // 程序组
    OptixProgramGroup raygen_prog_group_ = nullptr;
    OptixProgramGroup miss_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_prog_group_ = nullptr;
    
    // 设备指针
    CUdeviceptr d_params = 0;
};
} // namespace CubeDemo
