// include/graphics/fwd.h
#pragma once

// 该头文件非graphics模块的基类.
// 仅为向前声明使用
namespace CubeDemo {

struct BoundingSphere;
struct Vertex;
class Mesh;
class Shader;
namespace Graphics {
    class LODSystem;
}
}
using UnsignedArray = std::vector<unsigned>;
