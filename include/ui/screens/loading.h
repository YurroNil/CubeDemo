// include/ui/screens/loading.h
#pragma once

namespace CubeDemo::UI {
class LoadingScreen {
public:
    inline static bool s_Inited = false, s_isLoading = false;

    static void Init();
    static void Cleanup();
    static void Render(bool async_mode);
    static void StaticGraphic();
    static void DynamicGraphic();
};
} // namespace CubeDemo::UI
