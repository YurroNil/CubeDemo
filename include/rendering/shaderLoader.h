// include/rendering/shaderLoader.h

#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using string = std::string;

class ShaderLoader {
public:
    static string Load(const string& path);
};
