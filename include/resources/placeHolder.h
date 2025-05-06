// include/resources/placeHolder.h
#pragma once
#include "loaders/image.h"
#include "kits/glfw.h"
#include <iostream>

namespace CubeDemo {

// 占位符结构体
struct PlaceHolder : public Loaders::Texture
{
    static TexturePtr Create(const string& path, const string& type);
    static void ScheAsyncLoad(const string& path, const string& type, TexturePtr placeholder);
    static void FinalizeTex(TexturePtr placeholder, TexturePtr realTex);
    static void HandleLoadFailure(const string& path, const string& type,TexturePtr placeholder);
    static void ApplyTex(TexturePtr tex);
    static unsigned CreatePatterns();

};

}