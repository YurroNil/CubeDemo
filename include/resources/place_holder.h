// include/resources/place_holder.h
#pragma once
#include "loaders/image.h"
#include "resources/fwd.h"

namespace CubeDemo {

// 占位符结构体
struct PlaceHolder : public Loaders::Texture
{
    static TexturePtr Create(const string& path, const string& type);
};
}
