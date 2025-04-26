// include/resources/imageData.h
#pragma once

#include "threads/textureLoader.h"
#include "threads/resourceLoader.h"

namespace CubeDemo {

// 占位符结构体
struct PlaceHolder : public TextureLoader
{
    static TexturePtr Create(const string& path, const string& type);
    static void ScheAsyncLoad(const string& path, const string& type, TexturePtr placeholder);
    static void FinalizeTex(TexturePtr placeholder, TexturePtr realTex);
    static void HandleLoadFailure(const string& path, const string& type,TexturePtr placeholder);
    static void ApplyTex(TexturePtr tex);
    static unsigned CreatePatterns();

};

// 图像数据结构体（支持移动）
struct ImageData {

// 成员变量
    std::unique_ptr<unsigned char[]> data;
    int width = 0;
    int height = 0;
    int channels = 0;

// 方法
    static ImageDataPtr Load(const string& path);

// 显式禁止拷贝
    ImageData() = default;
    ImageData(ImageData&&) noexcept = default;
    ImageData& operator=(ImageData&&) noexcept = default;
    ImageData(const ImageData&) = delete;
    ImageData& operator=(const ImageData&) = delete;

};


}