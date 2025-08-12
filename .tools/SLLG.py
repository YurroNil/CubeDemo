# SLLG - Static Linking Library Generator
# 静态链接库生成器, CubeDemo项目工具链的工具之一
# version: 1.2 (2025.7.24)

import os
import json
import subprocess
from pathlib import Path

class SLLG:
    def __init__(self):
        self.configs = []  # 存储所有配置块
    
    def load_config(self, config_path="sllg_config.json"):
        """加载并解析配置文件"""
        print("\n\033[1;35m" + "=" * 60 + "\033[0m")
        print("\033[1;35m静态链接库生成器(SLLG)\033[0m")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.configs = json.load(f)
            
            # 只保留启用的配置块
            self.configs = [config for config in self.configs if config.get("is_running", True)]
            
            if not self.configs:
                print("\033[1;33m警告: 没有启用的配置块，程序将退出\033[0m")
                return False
            
            print(f"\n\033[1;32m找到 {len(self.configs)} 个启用的配置块!\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[1;31m错误: {str(e)}\033[0m")
            return False
    
    def process_config(self, config):
        """处理单个配置块"""
        # 解析配置数据
        library_name = config["library_name"]
        include_paths = config["paths"]["include"]
        source_dirs = config["paths"]["source_code"]
        cmd_prefix = config["arguments"]["cmd_prefix"]
        lib_suffix = config["arguments"]["lib_suffix"]
        output_dir = config["arguments"]["output"]
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 收集所有源文件
        source_files = self.collect_source_files(source_dirs)
        
        print("\n\033[1;35m" + "=" * 60 + "\033[0m")
        print(f"\033[1;35m开始处理库: {library_name}\033[0m")
        print(f"  库名称: {library_name}")
        print(f"  源文件数: {len(source_files)}")
        print(f"  输出目录: {output_dir}")
        
        # 编译所有源文件
        object_files = []
        if not self.compile_all_src_files(source_files, library_name, include_paths, cmd_prefix, lib_suffix, output_dir, object_files):
            return False
        
        # 创建静态库
        if not self.create_static_library(library_name, output_dir, object_files):
            return False
        
        return True
    
    def collect_source_files(self, source_dirs):
        """递归收集所有源文件(.c, .cpp)"""
        source_files = []
        for source_dir in source_dirs:
            # 扩展环境变量
            expanded_dir = os.path.expandvars(source_dir)
            if not os.path.exists(expanded_dir):
                print(f"\033[1;33m警告: 目录不存在 - {expanded_dir}\033[0m")
                continue
                
            # 递归查找源文件
            for ext in ["**/*.cpp", "**/*.c"]:
                source_files.extend(
                    [str(p) for p in Path(expanded_dir).glob(ext) if p.is_file()]
                )
        return source_files
    
    def generate_object_name(self, source_path, library_name, source_dirs):
        """生成目标文件名"""
        # 获取第一个源目录的父目录作为基础路径
        base_path = Path(os.path.expandvars(source_dirs[0])).parent
        
        # 获取相对路径
        try:
            relative_path = Path(source_path).relative_to(base_path)
        except ValueError:
            # 如果不在同一个根目录下，使用绝对路径的hash
            path_hash = abs(hash(source_path)) % (10**8)
            return f"{library_name}_{path_hash}.o"
        
        # 处理父目录路径
        if relative_path.parent != Path('.'):
            parent_str = str(relative_path.parent).replace('\\', '_').replace('/', '_')
            object_name = f"{library_name}_{parent_str}_{relative_path.stem}.o"
        else:
            object_name = f"{library_name}_{relative_path.stem}.o"
        
        return object_name
    
    def build_compile_command(self, source_path, object_path, include_paths, cmd_prefix, lib_suffix):
        """构建编译命令"""
        # 处理包含路径
        include_flags = "".join([f' -I"{os.path.expandvars(path)}"' for path in include_paths])
        
        # 处理库后缀
        lib_suffix_str = "".join([f" {suffix}" for suffix in lib_suffix])
        
        # 构建完整命令
        return (f'{cmd_prefix} "{source_path}"{include_flags} '
                f'-o "{object_path}"{lib_suffix_str}')
    
    def compile_all_src_files(self, source_files, library_name, include_paths, cmd_prefix, lib_suffix, output_dir, object_files):
        """编译所有源文件"""
        print("\n\033[1;35m阶段 1: 编译源文件\033[0m")
        
        try:
            for source_path in source_files:
                # 生成目标文件名
                object_name = self.generate_object_name(source_path, library_name, include_paths)
                object_path = os.path.join(output_dir, object_name)
                object_files.append(object_path)
                
                # 确保目标目录存在
                Path(object_path).parent.mkdir(parents=True, exist_ok=True)
                
                # 构建并执行编译命令
                compile_cmd = self.build_compile_command(
                    source_path, object_path, include_paths, cmd_prefix, lib_suffix
                )
                print(f"\n\033[1;34m执行命令: \033[0m{compile_cmd}")
                
                # 执行编译命令
                result = subprocess.run(compile_cmd, shell=True)
                if result.returncode != 0:
                    print(f"\033[1;31m编译失败: {source_path}\033[0m")
                    return False
            return True
            
        except Exception as e:
            print(f"\033[1;31m错误: {str(e)}\033[0m")
            return False
    
    def create_static_library(self, library_name, output_dir, object_files):
        """创建静态库"""
        print("\n\033[1;35m阶段 2: 创建静态库\033[0m")
        
        try:
            # 构建静态库路径
            library_path = os.path.join(output_dir, f"lib{library_name}.a")
            
            # 构建打包命令
            archive_cmd = f'ar.exe rcs "{library_path}"'
            for obj in object_files:
                archive_cmd += f' "{obj}"'
            
            # 执行打包命令
            print(f"\n\033[1;34m执行命令: \033[0m{archive_cmd}")
            result = subprocess.run(archive_cmd, shell=True)
            
            if result.returncode != 0:
                print("\033[1;31m静态库创建失败\033[0m")
                return False
            
            # 显示成功信息
            self.pop_notification(library_path, len(object_files))
            return True
            
        except Exception as e:
            print(f"\033[1;31m错误: {str(e)}\033[0m")
            return False
    
    def pop_notification(self, library_path, object_count):
        """任务完成通知"""
        print("\n\033[1;35m" + "=" * 60 + "\033[0m")
        print("\033[1;32m静态库创建成功!\033[0m")
        print(f"  库路径: {library_path}")
        print(f"  包含目标文件: {object_count} 个")
        print("=" * 60)
    
    def run(self):
        """运行整个流程"""
        if not self.load_config():
            return False
        
        # 处理所有启用的配置块
        success = True
        for config in self.configs:
            if not self.process_config(config):
                success = False
                print(f"\033[1;31m处理配置块失败: {config['library_name']}\033[0m")
            else:
                print(f"\033[1;32m成功处理配置块: {config['library_name']}\033[0m")
        
        return success


if __name__ == "__main__":
    generator = SLLG()
    if generator.run():
        exit(0)
    else:
        exit(1)
