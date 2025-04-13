// include/graphics/shaderLoader.h

#pragma once
#include "utils/stringsKits.h"
namespace CubeDemo {

class ShaderLoader {
public:
    static string Load(const string& path);
};

}