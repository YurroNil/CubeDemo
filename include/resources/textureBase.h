// include/resources/textureBase.h
#pragma once
#include "pch.h"

namespace CubeDemo {
    // 前向声明
    class Texture;
    using TexturePtr = std::shared_ptr<Texture>;
    using TexPtrHashMap = std::unordered_map<string, std::weak_ptr<Texture>>;

}