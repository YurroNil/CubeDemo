// include/resources/fwd.h
#pragma once

namespace CubeDemo {
    // 前向声明
    struct PlaceHolder; class model; class Texture;

    // 别名
    using TexturePtr = std::shared_ptr<Texture>;
    using TexPtrHashMap = std::unordered_map<string, std::weak_ptr<Texture>>;

}