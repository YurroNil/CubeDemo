// include/graphics/shaderLoader.h

#pragma once
#include "utils/stringsKits.h"
namespace CubeDemo {

class ShaderLoader {
public:
    static string Load(const string& path);
    static const string s_vshPath;
    static const string s_fshPath;
};

}