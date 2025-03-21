// include/rendering/shaderLoader.h

#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


class ShaderLoader {
public:
    static std::string Load(const std::string& path);
};
