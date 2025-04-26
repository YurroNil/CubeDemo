// 新增gl_tracker.cpp
#include <unordered_map>
#include "utils/glfwKits.h"
#include "utils/stringsKits.h"
#include <mutex>
#include <iostream>

namespace CubeDemo {
    static std::unordered_map<GLuint, string> s_GLResources;
    static std::mutex s_GLTrackerMutex;

    void TrackGLObj(GLuint id, const string& type) {
        std::lock_guard lock(s_GLTrackerMutex);
        s_GLResources[id] = type;
    }

    void UntrackGLObj(GLuint id) {
        std::lock_guard lock(s_GLTrackerMutex);
        s_GLResources.erase(id);
    }

    void DumpGLResources() {
        std::lock_guard lock(s_GLTrackerMutex);
        std::cout << "=== 存活OpenGL资源 ===" << std::endl;
        for(const auto& [id, type] : s_GLResources) {
            std::cout << type << " #" << id << std::endl;
        }
    }
}
