// include/graphics/lod.h

#pragma once
#include "graphics/base.h"
#include "utils/msaConv.h"

using MeshArray = std::vector<CubeDemo::Mesh>;
namespace CubeDemo::Graphics {

class LODLevel {
public:
    // 显式声明移动操作
    LODLevel(LODLevel&& other) noexcept;
    LODLevel& operator=(LODLevel&& other) noexcept;
    
    // 删除拷贝操作
    LODLevel(const LODLevel&) = delete;
    LODLevel& operator=(const LODLevel&) = delete;

    LODLevel(MeshArray meshes, float transition_dist, bool ready = false);
    
    const MeshArray& GetMeshes() const noexcept;
    float GetTransDistance() const noexcept;
    bool IsReady() const noexcept;
    void MarkReady() noexcept;

private:
    MeshArray m_Meshes;
    float m_TransDistance;
    std::atomic<bool> m_IsReady;
};

using LodLevelArray = std::vector<LODLevel>;
class LODSystem {
public:
    // 网格简化算法别名
    using SimplifyAlg = std::function<Mesh(const Mesh&, float)>;

    // LOD系统初始化
    void Init(MeshArray&& base_meshes,
        float bound_rad,
        const std::vector<float>& ratios = {0.3f, 0.6f},
        SimplifyAlg algorithm = simplify_mesh
    );

    // 添加预生成的LOD层级
    void AddPrebuiltLevel(LODLevel&& level);
    
    // 核心选择算法
    const LODLevel& SelectLevel(const vec3& anchor_point, const vec3& observer_pos) const;
    
    // 异步生成接口
    void GenLodHierarchy_async(
        const LODLevel& base_level,
        const std::vector<float>& simp_factors,
        SimplifyAlg algorithm = simplify_mesh
    );

    // 同步生成接口
    void GenLodHierarchy_sync(
        const LODLevel& base_level,
        const std::vector<float>& simp_factors,
        SimplifyAlg algorithm = simplify_mesh
    );

    // 层级访问
    const LODLevel& GetLevelByIndex(size_t index) const;
    size_t GetLevelCount() const noexcept;

    const LodLevelArray& GetlevelArray() const;

private:
    mutable std::mutex m_LevelsMutex;
    LodLevelArray m_Levels;

};

} // namespace CubeDemo::Graphics
