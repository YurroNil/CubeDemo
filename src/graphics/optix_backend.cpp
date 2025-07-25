// src/graphics/optix_backend.cpp
#include "pch.h"

#include <cuda.h>            // 添加CUDA驱动API头文件
#include <cuda_gl_interop.h> // CUDA-OpenGL互操作
#include "graphics/optix_backend.h"
#include <glm/gtc/type_ptr.hpp>

// 解决undefined reference to `g_optixFunctionTable_105`的问题.
extern "C" { OptixFunctionTable g_optixFunctionTable_105 = {}; }

namespace CubeDemo {

OptixBackend& OptixBackend::GetInstance() {
    static OptixBackend instance;
    return instance;
}

OptixBackend::OptixBackend() {}

OptixBackend::~OptixBackend() {
    Shutdown();
}

void OptixBackend::Init(int width, int height) {
    if (context_) return;
    
    width_ = width;
    height_ = height;
    
    // 初始化 OptiX
    OptixResult initResult = optixInit();
    if (initResult != OPTIX_SUCCESS) {
        std::cerr << "OptiX initialization failed: " << initResult << std::endl;
        throw std::runtime_error("OptiX initialization failed");
    }
    
    // 初始化 CUDA
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // 获取当前CUDA上下文
    CUcontext cuContext = nullptr;
    CUresult cuResult = cuCtxGetCurrent(&cuContext);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "Failed to get CUDA context: " << cuResult << std::endl;
        return;
    }
    
    // 创建OptiX上下文
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context_));
    
    // 创建 OpenGL 纹理
    glGenTextures(1, &output_texture_);
    glBindTexture(GL_TEXTURE_2D, output_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // 注册CUDA资源
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &cuda_texture_resource_, 
        output_texture_, 
        GL_TEXTURE_2D, 
        cudaGraphicsRegisterFlagsWriteDiscard
    ));
    
    // 创建OptiX管道
    CreatePipeline();
}

void OptixBackend::Shutdown() {
    if (d_triangles) {
        cudaFree(reinterpret_cast<void*>(d_triangles));
        d_triangles = 0;
    }
    if (d_materials) {
        cudaFree(reinterpret_cast<void*>(d_materials));
        d_materials = 0;
    }
    free_gas(gas_);
    
    if (context_) {
        OPTIX_CHECK(optixDeviceContextDestroy(context_));
        context_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (cuda_texture_resource_) {
        cudaGraphicsUnregisterResource(cuda_texture_resource_);
        cuda_texture_resource_ = nullptr;
    }
    if (output_texture_) {
        glDeleteTextures(1, &output_texture_);
        output_texture_ = 0;
    }

    // 清理管道相关资源
    if (pipeline_) {
        OPTIX_CHECK(optixPipelineDestroy(pipeline_));
        pipeline_ = nullptr;
    }
    if (module_) {
        OPTIX_CHECK(optixModuleDestroy(module_));
        module_ = nullptr;
    }
    if (raygen_prog_group_) {
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
        raygen_prog_group_ = nullptr;
    }
    if (miss_prog_group_) {
        OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
        miss_prog_group_ = nullptr;
    }
    if (hitgroup_prog_group_) {
        OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
        hitgroup_prog_group_ = nullptr;
    }
}

void OptixBackend::UploadScene(
    const std::vector<OptixTriangle>& triangles,
    const std::vector<OptixMaterial>& materials
) {
    // 清理之前的资源
    if (d_triangles) cudaFree(reinterpret_cast<void*>(d_triangles));
    if (d_materials) cudaFree(reinterpret_cast<void*>(d_materials));
    free_gas(gas_);
    
    // 设置三角形数量
    triangle_count_ = triangles.size();
    
    // 上传三角形数据
    if (!triangles.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles), triangles.size() * sizeof(OptixTriangle)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_triangles), 
            triangles.data(),
            triangles.size() * sizeof(OptixTriangle), 
            cudaMemcpyHostToDevice
        ));
    }
    
    // 上传材质数据
    if (!materials.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materials), materials.size() * sizeof(OptixMaterial)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_materials), 
            materials.data(),
            materials.size() * sizeof(OptixMaterial), 
            cudaMemcpyHostToDevice
        ));
    }
    
    // 构建加速结构
    BuildAccelerationStructure();
}

void OptixBackend::UpdateCamera(
    const glm::vec3& position,
    const glm::vec3& lookAt,
    const glm::vec3& up,
    float fov,
    float aspect
) {
    camera_params_.eye = make_float3(position.x, position.y, position.z);
    camera_params_.lookat = make_float3(lookAt.x, lookAt.y, lookAt.z);
    camera_params_.up = make_float3(up.x, up.y, up.z);
    camera_params_.fov = fov;
    camera_params_.aspect = aspect;
}

void OptixBackend::Render() {
    if (!context_) return;
    
    // 映射CUDA资源
    cudaArray_t cuda_array;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_texture_resource_, stream_));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_texture_resource_, 0, 0));
    
    // 设置启动参数
    Params params;
    params.handle = gas_.handle;
    params.image = cuda_array;
    params.width = width_;
    params.height = height_;
    params.eye = camera_params_.eye;
    params.lookat = camera_params_.lookat;
    params.up = camera_params_.up;
    params.fov = camera_params_.fov;
    params.aspect = camera_params_.aspect;
    params.triangles = reinterpret_cast<OptixTriangle*>(d_triangles);
    params.materials = reinterpret_cast<OptixMaterial*>(d_materials);
    
    // 上传参数到设备
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_params),
        &params,
        sizeof(Params),
        cudaMemcpyHostToDevice
    ));
    
    // 启动OptiX管线
    OPTIX_CHECK(optixLaunch(
        pipeline_, 
        stream_, 
        d_params,
        sizeof(Params), 
        &sbt_, 
        width_, 
        height_, 
        1
    ));
    
    // 解映射资源
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_texture_resource_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ============== 内部实现 ==============

static string read_ptx_file(const string& filename) {
    // 使用相对路径或绝对路径
    string full_path = PTX_PATH + filename;
    std::ifstream file(full_path, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error opening PTX file: " << full_path << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void OptixBackend::CreatePipeline() {
    // 读取 PTX 文件
    string ptx = read_ptx_file("raytracing.ptx");
    if (ptx.empty()) {
        throw std::runtime_error("Failed to load PTX file");
    }
    
    // 创建模块编译选项
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = 50;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    
    // 创建管道编译选项
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    
    // 创建模块
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    OPTIX_CHECK(optixModuleCreate(
        context_,
        &module_compile_options,
        &pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &module_
    ));
    
    // 创建程序组
    OptixProgramGroupOptions program_group_options = {};
    
    // 光线生成程序组
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module_;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__main";
    
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &raygen_prog_group_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        &raygen_prog_group_
    ));
    
    // Miss 程序组
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module_;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
    
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &miss_prog_group_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        &miss_prog_group_
    ));
    
    // Hitgroup 程序组
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module_;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        context_,
        &hitgroup_prog_group_desc,
        1,
        &program_group_options,
        log,
        &sizeof_log,
        &hitgroup_prog_group_
    ));
    
    // 链接管道
    OptixProgramGroup program_groups[] = {
        raygen_prog_group_,
        miss_prog_group_,
        hitgroup_prog_group_
    };
    
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context_,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups)/sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline_
    ));
    
    // 设置着色器绑定表 (SBT)
    SetupSBT();
    
    // 分配参数缓冲区
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
}

void OptixBackend::SetupSBT() {
    // Raygen 记录
    RaygenRecord raygen_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &raygen_record));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.raygenRecord), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(sbt_.raygenRecord),
        &raygen_record,
        sizeof(RaygenRecord),
        cudaMemcpyHostToDevice
    ));
    
    // Miss 记录
    MissRecord miss_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_, &miss_record));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.missRecordBase), sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(sbt_.missRecordBase),
        &miss_record,
        sizeof(MissRecord),
        cudaMemcpyHostToDevice
    ));
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    
    // Hitgroup 记录
    HitgroupRecord hitgroup_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &hitgroup_record));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_.hitgroupRecordBase), sizeof(HitgroupRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(sbt_.hitgroupRecordBase),
        &hitgroup_record,
        sizeof(HitgroupRecord),
        cudaMemcpyHostToDevice
    ));
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt_.hitgroupRecordCount = 1;
}

void OptixBackend::BuildAccelerationStructure() {
    if (triangle_count_ == 0) {
        std::cerr << "Warning: No triangles to build acceleration structure" << std::endl;
        return;
    }
    
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    build_input.triangleArray.vertexBuffers = &d_triangles;
    build_input.triangleArray.numVertices = static_cast<uint32_t>(triangle_count_);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(OptixTriangle);
    build_input.triangleArray.indexBuffer = 0;
    build_input.triangleArray.numIndexTriplets = 0;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    
    // 设置几何标志
    unsigned int geometry_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    build_input.triangleArray.flags = geometry_flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    // 构建 GAS
    gas_ = build_gas(context_, build_input, accel_options);
}
} // namespace CubeDemo
