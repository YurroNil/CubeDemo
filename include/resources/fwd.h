// include/resources/fwd.h
#pragma once

namespace CubeDemo {
    // 前向声明
    struct PlaceHolder; struct Vertex;
    class Mesh; class Model; class Texture; class Material;

    // 别名
    using TexturePtr = std::shared_ptr<Texture>;
    using MaterialPtr = std::shared_ptr<Material>;
    using TexPtrHashMap = std::unordered_map<string, std::weak_ptr<Texture>>;
}  // namespace
