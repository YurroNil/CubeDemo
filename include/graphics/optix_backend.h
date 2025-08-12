// include/graphics/optix_backend.h
#pragma once
#include "graphics/optix_utils.h"

namespace CubeDemo {

#include "graphics/optix_structures.h"
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
    
    cudaGraphicsResource_t GetOutputTextureResource() const { return cuda_texture_resource_; }
    unsigned int GetOutputTexture() const { return output_texture_; }

private:
    OptixBackend();
    ~OptixBackend();
    
    void CreatePipeline();
    void SetupSBT();
    void BuildAccelerationStructure();
    void ResetRayStates();
    
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

    // 添加输出缓冲区
    float4* d_output = nullptr;
    int material_count_ = 0;
    curandState* d_rand_state_ = nullptr;
    RayState* d_ray_states_ = nullptr;
    std::vector<OptixTriangle> host_triangles_;
};
} // namespace CubeDemo
