1. 类说明

Mesh类
    实现了基于ModelData的网格加载，自动绑定VAO/VBO
    包含基本的绘制方法，动态计算顶点数量
    TODO: 添加索引缓冲对象(IBO)支持以优化渲染性能

ModelLoader类
    从JSON文件加载模型数据（顶点、着色器路径等）
    包含基本的JSON验证逻辑
    TODO: 添加模型缓存机制避免重复加载

Renderer类
    管理OpenGL状态（深度测试、清屏等）
    实现基本的渲染管线提交
    TODO: 添加多光源支持、阴影映射等高级渲染特性

TextRenderer类
    使用FreeType实现文本渲染
    支持动态生成字形纹理
    TODO: 可添加文本缓存机制（如使用纹理图集）

TextureManager类
    统一管理纹理资源


2. 源代码说明(h/cpp/c)

*core/camera
    作用：摄像机控制器
    功能：
    管理视图矩阵(view matrix)
    实现摄像机移动（WASD/方向键）
    处理鼠标视角旋转
    计算投影矩阵(projection matrix)
    提供摄像机位置/方向查询接口


*core/fpsCounter
    作用：性能监控器
    功能：
    计算每秒帧数(FPS)
    统计渲染时间
    控制台输出性能指标
    可能集成帧率限制器


*inputHandler
    作用：输入管理系统
    功能：
    监听键盘/鼠标事件
    绑定控制映射（如ESC退出，F3打开调试面板）
    提供输入状态查询接口
    处理窗口焦点事件


*rendering/mesh
    作用：网格对象管理器
    功能：
    管理VAO/VBO/EBO
    存储顶点数据/索引数据
    提供绘制接口（Draw()方法）
    支持动态/静态网格类型


*rendering/modelLoader
    作用：模型加载器
    功能：
    解析cube.json等模型文件
    加载顶点/纹理坐标/法线数据
    自动生成缓冲区对象
    关联着色器程序


*rendering/shader
    作用：着色器管理器
    功能：
    编译GLSL着色器
    管理着色器程序对象
    设置uniform变量
    错误检查与日志输出


*rendering/textureLoader
    作用：纹理加载器
    功能：
    加载PNG/JPG等图片文件
    生成纹理对象
    支持纹理参数配置
    管理纹理内存


*renderer/main
    作用：主渲染器
    功能：
    初始化OpenGL上下文
    管理渲染队列
    执行渲染循环
    处理帧缓冲操作
    集成后处理效果
