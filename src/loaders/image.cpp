// src/loaders/image.cpp
#include "pch.h"
#include "loaders/image.h"
#include "loaders/progress_tracker.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace CubeDemo {

ImagePtr IL::Load(const string& path) {

    // 更新进度：开始加载图像
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::TEXTURE_IO, 
        path, 0.0f
    );

    auto data = std::make_shared<IL>();

    stbi_set_flip_vertically_on_load(false);


    // 分阶段更新进度
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::TEXTURE_IO, 
        path, 
        0.5f
    );

    data->data.reset(
        stbi_load(
            path.c_str(),
            &data->width,
            &data->height,
            &data->channels,
            0
        )
    );
    if(!data->data) throw std::runtime_error(stbi_failure_reason());
    
    // 更新IO进度完成
    ProgressTracker::Get().UpdateProgress(
        ProgressTracker::TEXTURE_IO,
        path, 1.0f
    );
    return data;
}
}   // namespace CubeDemo
