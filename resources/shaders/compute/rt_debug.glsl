// resources/shaders/compute/ray_tracing.glsl
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform image2D outputImage;

uniform struct Camera {
    vec3 position;
    vec3 front;
    vec3 up;
    vec3 right;
} uCamera;

uniform float time;

// 球体定义
struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float specular;
    float reflectivity;
};

#define NUM_SPHERES 5
const Sphere spheres[NUM_SPHERES] = Sphere[NUM_SPHERES](
    Sphere(vec3(0.0, 0.0, -5.0), 1.0, vec3(1.0, 0.0, 0.0), 32.0, 0.3),   // 红色球
    Sphere(vec3(-2.0, 1.0, -6.0), 1.0, vec3(0.0, 1.0, 0.0), 64.0, 0.5),  // 绿色球
    Sphere(vec3(2.0, -1.0, -7.0), 1.0, vec3(0.0, 0.0, 1.0), 128.0, 0.7), // 蓝色球
    Sphere(vec3(0.0, -101.0, -5.0), 100.0, vec3(0.8, 0.8, 0.8), 16.0, 0.2), // 地面
    Sphere(vec3(0.0, 5.0, -10.0), 1.5, vec3(1.0, 1.0, 0.8), 256.0, 0.1)  // 光源
);

// 光线与球体求交
float intersectSphere(Sphere s, vec3 rayOrigin, vec3 rayDir) {
    vec3 oc = rayOrigin - s.center;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(oc, rayDir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return -1.0;
    }
    
    return (-b - sqrt(discriminant)) / (2.0 * a);
}

// 场景求交
bool intersectScene(vec3 rayOrigin, vec3 rayDir, out int hitIndex, out float minT) {
    minT = 10000.0;
    hitIndex = -1;
    
    for (int i = 0; i < NUM_SPHERES; i++) {
        float t = intersectSphere(spheres[i], rayOrigin, rayDir);
        if (t > 0.0001 && t < minT) {
            minT = t;
            hitIndex = i;
        }
    }
    
    return hitIndex != -1;
}

// 计算法向量
vec3 calculateNormal(Sphere sphere, vec3 point) {
    return normalize(point - sphere.center);
}

// 简单随机数生成
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 光线追踪主函数
vec3 traceRay(vec3 rayOrigin, vec3 rayDir) {
    vec3 color = vec3(0.0);
    vec3 attenuation = vec3(1.0);
    
    for (int bounce = 0; bounce < 3; bounce++) {
        int hitIndex;
        float t;
        
        if (!intersectScene(rayOrigin, rayDir, hitIndex, t)) {
            // 天空颜色
            float t = 0.5 * (rayDir.y + 1.0);
            vec3 skyColor = (1.0 - t) * vec3(0.1, 0.1, 0.3) + t * vec3(0.5, 0.7, 1.0);
            color += attenuation * skyColor;
            break;
        }
        
        Sphere hitSphere = spheres[hitIndex];
        vec3 hitPoint = rayOrigin + t * rayDir;
        vec3 normal = calculateNormal(hitSphere, hitPoint);
        
        // 光源位置
        vec3 lightPos = spheres[4].center;
        vec3 lightDir = normalize(lightPos - hitPoint);
        
        // 阴影检测
        vec3 shadowOrigin = hitPoint + normal * 0.001;
        int shadowIndex;
        float shadowT;
        bool inShadow = intersectScene(shadowOrigin, lightDir, shadowIndex, shadowT);
        
        // 基础光照
        float diff = max(dot(normal, lightDir), 0.0);
        
        // 镜面反射
        vec3 viewDir = normalize(rayOrigin - hitPoint);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), hitSphere.specular);
        
        // 光照计算（排除光源本身）
        vec3 lighting = vec3(0.0);
        if (hitIndex != 4) {
            if (!inShadow || shadowIndex == 4) {
                lighting = diff * hitSphere.color + vec3(0.8) * spec;
            } else {
                lighting = diff * hitSphere.color * 0.3;
            }
        }
        
        // 自发光（光源）
        if (hitIndex == 4) {
            lighting = hitSphere.color * 5.0;
        }
        
        color += attenuation * lighting;
        
        // 反射
        if (hitSphere.reflectivity > 0.0) {
            attenuation *= hitSphere.reflectivity;
            rayDir = reflect(rayDir, normal);
            rayOrigin = hitPoint + normal * 0.001;
        } else {
            break;
        }
    }
    
    return color;
}

void main() {
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(storePos) + 0.5) / vec2(imageSize(outputImage));
    uv = uv * 2.0 - 1.0;
    
    // 相机设置
    float aspectRatio = float(imageSize(outputImage).x) / float(imageSize(outputImage).y);
    
    // 使用传入的相机方向向量
    vec3 rayDir = normalize(uCamera.front + uv.x * uCamera.right * aspectRatio + uv.y * uCamera.up);
    
    // 执行光线追踪
    vec3 color = traceRay(uCamera.position, rayDir);
    
    // 添加一些噪声模拟胶片颗粒
    float noise = 0.05 * rand(vec2(storePos) + time);
    color += vec3(noise);
    
    // 色调映射
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2)); // Gamma校正
    
    // 输出颜色
    imageStore(outputImage, storePos, vec4(color, 1.0));
}
