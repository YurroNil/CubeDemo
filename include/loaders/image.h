// include/loaders/image.h
#pragma once

#include "loaders/texture.h"
#include "loaders/resource.h"

using millisec = std::chrono::milliseconds;
using IL = CubeDemo::Loaders::Image;

namespace CubeDemo {

// 图像数据结构体（支持移动）
class Loaders::Image {
public:
// 成员变量
    std::unique_ptr<unsigned char[]> data;

    int width = 0;
    int height = 0;
    int channels = 0;
// 方法
    static ImagePtr Load(const string& path);
// 显式禁止拷贝
    Image() = default;
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
};


}