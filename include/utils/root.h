// include/root.h
#pragma once

// INCLUDE
#include <vector>
#include <functional>
#include <string>
#include <codecvt>  // 用于 wstring_convert
#include <locale>

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"


// USING
using string = std::string;
using wstring = std::wstring;

using vec3 = glm::vec3;
using vec4 = glm::vec4;
using mat4 = glm::mat4;
