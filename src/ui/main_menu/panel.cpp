// src/ui/main_menu/panel.cpp
#include "pch.h"
#include "ui/main_menu/_all.h"
#include "loaders/font.h"
#include "loaders/image.h"
#include "utils/time.h"
#include "utils/font_defines.h"
#include "managers/scene.h"

using TLS = CubeDemo::Texture::LoadState;

namespace CubeDemo { extern Managers::SceneMng* SCENE_MNG; }

namespace CubeDemo::UI {

void MainMenuPanel::Init() {
    glfwSetInputMode(WINDOW::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // 从场景管理器加载场景列表
    LoadScenelistFrom();
    // 加载场景预览图
    LoadScenePreviews();
    m_greeting = Utils::Time::get_time_greeting();
}

void MainMenuPanel::LoadScenelistFrom() {

    // 清空现有场景列表
    MainMenuBase::m_sceneList.clear();
    
    // 从场景管理器获取所有场景
    const auto& scenes = SCENE_MNG->GetAllScenes();
    
    // 构建场景列表
    for (const auto& [sceneID, scene] : scenes) {
        const auto& info = scene->GetSceneInfo();
    
        MainMenuBase::SceneItem item;
        item.name = info.name;
        item.description = info.description;
        item.author = info.author;
        item.icon = info.icon;
        item.previewImage = info.previewImage;
        item.path = info.resourcePath; // 使用资源路径作为标识

        MainMenuBase::m_sceneList.push_back(item);
    }
}

void MainMenuPanel::LoadScenePreviews() {
    // 从场景管理器获取所有场景
    const auto& scenes = SCENE_MNG->GetAllScenes();
    
    for (auto& sceneItem : MainMenuBase::m_sceneList) {
        // 查找对应的场景信息
        bool found = false;
        Scenes::SceneInfo sceneInfo;
        
        for (const auto& [id, scene] : scenes) {
            if (scene->GetSceneInfo().resourcePath != sceneItem.path) continue;
            
            sceneInfo = scene->GetSceneInfo();
            found = true;
            break;
        }
        
        if (!found) continue;
        
        // 尝试加载预览图
        if (!sceneInfo.previewImage.empty()) {
            try {
                // 构建完整路径
                string fullPath = sceneInfo.resourcePath + "/" + sceneInfo.previewImage;
                
                // 使用同步加载纹理
                sceneItem.preview = TL::LoadSync(fullPath, "diffuse");
                
                // 如果加载失败，使用占位符
                if (!sceneItem.preview || sceneItem.preview->State.load() != TLS::Ready) {
                    CreatePlaceholderTexture(sceneItem, sceneInfo.id);
                }
            } catch (...) {
                // 加载失败，使用占位符
                CreatePlaceholderTexture(sceneItem, sceneInfo.id);
            }
        // 没有预览图路径，使用占位符
        } else CreatePlaceholderTexture(sceneItem, sceneInfo.id);
    }
}

void MainMenuPanel::CreatePlaceholderTexture(SceneItem& sceneItem, const string& sceneId) {
    // 创建1x1的纯色图像数据
    auto imageData = std::make_shared<IL>();
    imageData->width = 1;
    imageData->height = 1;
    imageData->channels = 4;
    
    // 生成随机颜色
    unsigned char* data = new unsigned char[4]{
        static_cast<unsigned char>(rand() % 200 + 55),
        static_cast<unsigned char>(rand() % 200 + 55),
        static_cast<unsigned char>(rand() % 200 + 55),
        255
    };
    
    imageData->data = std::unique_ptr<unsigned char[]>(data);
    
    // 使用纹理加载器创建纹理
    sceneItem.preview = TL::CreateFromData(imageData, "placeholder_" + sceneId, "diffuse");
}

void MainMenuPanel::Render() {
    if (!s_isMainMenuPhase) return;
    
    // 全屏主菜单背景
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    
    // 增加窗口内边距
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(40, 40));
    
    if (ImGui::Begin("MainMenu", &s_isMainMenuPhase, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove | 
        ImGuiWindowFlags_NoScrollbar | 
        ImGuiWindowFlags_NoCollapse | 
        ImGuiWindowFlags_NoBringToFrontOnFocus | 
        ImGuiWindowFlags_NoBackground | 
        ImGuiWindowFlags_NoScrollWithMouse
    )) {
        // 绘制渐变背景
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const ImVec2 p_min = viewport->Pos;
        const ImVec2 p_max = ImVec2(viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y);
        
        draw_list->AddRectFilledMultiColor(
            p_min, p_max,
            ImColor(0.10f, 0.10f, 0.10f, 0.95f),
            ImColor(0.12f, 0.12f, 0.12f, 0.95f),
            ImColor(0.08f, 0.08f, 0.08f, 0.95f),
            ImColor(0.10f, 0.10f, 0.10f, 0.95f)
        );

        // 添加星空效果
        static bool starsInited = false;
        static std::vector<ImVec2> starPositions;
        static std::vector<float> starSizes;
        
        if (!starsInited) {
            for (int i = 0; i < 100; i++) {
                starPositions.push_back(ImVec2(
                    rand() % (int)viewport->Size.x,
                    rand() % (int)viewport->Size.y
                ));
                starSizes.push_back(0.5f + (rand() % 100) / 200.0f);
            }
            starsInited = true;
        }
        
        for (int i = 0; i < 100; i++) {
            draw_list->AddCircleFilled(
                ImVec2(starPositions[i].x, starPositions[i].y), 
                starSizes[i], ImColor(1.0f, 1.0f, 1.0f, 0.5f)
            );
        }
        
        // 渲染菜单各部分
        MainMenu::Menubar::Render();
        MainMenu::TitleSection::Render();
        
        // 计算两个窗口的总宽度和起始位置
        float totalWidth = ImGui::GetWindowWidth() * (SELECTION_WIDTH_RATIO + PREVIEW_WIDTH_RATIO) + WINDOW_SPACING;
        float startX = (ImGui::GetWindowWidth() - totalWidth) * 0.5f;
        float startY = ImGui::GetWindowHeight() * 0.4f;
        
        // 设置选择场景窗口位置
        ImGui::SetCursorPos(ImVec2(startX, startY));
        MainMenu::SceneSelection::Render();
        
        // 设置预览窗口位置
        ImGui::SetCursorPos(ImVec2(startX + ImGui::GetWindowWidth() * SELECTION_WIDTH_RATIO + WINDOW_SPACING, startY));
        
        MainMenu::PreviewPanel::Render();
        
        MainMenu::Bottombar::Render();
    }
    ImGui::End();

    ImGui::PopStyleVar(); // 弹出窗口内边距设置
}
} // namespace CubeDemo::UI
