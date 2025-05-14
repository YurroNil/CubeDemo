// include/utils/MsaConv.h
#pragma once

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

#include "graphics/mesh.h"

namespace CubeDemo {

// 修正属性设置，确保各元素使用正确的属性标志
struct FinalTraits : public OpenMesh::DefaultTraits {
    
    // 顶点属性：法线, 二维纹理坐标, 状态
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::TexCoord2D | OpenMesh::Attributes::Status);

    // 面属性：法线
    FaceAttributes(OpenMesh::Attributes::Normal);
    // 边属性：状态
    EdgeAttributes(OpenMesh::Attributes::Status);
    // 半边属性：前驱半边和状态
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
};

namespace Graphics {

void conv_to_openMesh(const CubeDemo::Mesh& src, OpenMesh::TriMesh_ArrayKernelT<FinalTraits>& trimesh);

CubeDemo::Mesh conv_from_openMesh(const OpenMesh::TriMesh_ArrayKernelT<FinalTraits>& trimesh, const CubeDemo::Mesh& original);

CubeDemo::Mesh simplify_mesh(const CubeDemo::Mesh& src, float simplify_ratio);

}   // namespace CubeDemo::Graphics

}   // namespace CubeDemo
