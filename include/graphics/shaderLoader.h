// include/graphics/shaderLoader.h

#pragma once
#include "utils/root.h"
#include "utils/streams.h"

class ShaderLoader {
public:
    static string Load(const string& path);
    static const string s_vshPath;
    static const string s_fshPath;
};
