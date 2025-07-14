// include/resources/material.h
#pragma once
#include "resources/fwd.h"

namespace CubeDemo {

class Material {
public:
    vec3 diffuse, specular, emission;
    float shininess, opacity;
};
}   // namespace CubeDemo
