//res/shaders/core/lit.vsh
#version 330 core
// 顶点属性输入（从VBO获取）
layout (location = 0) in vec3 aPos;    // 顶点位置
layout (location = 1) in vec3 aNormal; // 顶点法线
 
// 输出到片段着色器的变量
out vec3 Normal;      // 变换后的法线向量
out vec3 FragPos;     // 变换后的顶点位置
 
// 统一矩阵（从外部传入）
uniform mat4 model;     // 模型矩阵
uniform mat4 view;      // 观察矩阵
uniform mat4 projection; // 投影矩阵
 
void main() {
    // ========== 世界空间坐标计算 ==========
    // 将顶点位置变换到世界空间（model矩阵包含缩放/旋转/平移）
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // ========== 法线变换 ==========
    // 计算法线矩阵：先转置模型矩阵的逆矩阵，再取3x3部分
    // 注意：仅当模型矩阵包含非均匀缩放时需要，若只有旋转/平移可简化为转置
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    // ========== 裁剪空间坐标计算 ==========
    // 标准的MVP变换：投影 * 观察 * 模型 * 顶点位置
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}