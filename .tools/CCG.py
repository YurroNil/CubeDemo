# CCG - Compile Command Generator
# 编译命令生成器, CubeDemo项目工具链的工具之一
# version: 1.2 (2025.7.24)

import os
import json
import subprocess
from pathlib import Path

class CCG:
    def __init__(self):
        self.configs = []  # 存储所有配置块
    
    def load_config(self, config_path="ccg_config.json"):
        """加载并解析配置文件"""
        print("\n\033[1;35m" + "=" * 60 + "\033[0m")
        print("\033[1;35m编译命令生成器(CCG) - 启动\033[0m")
        
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
    
    def validate_config(self, config):
        """验证配置块结构"""
        required_keys = {
            "paths": ["include", "lib", "source_code"],
            "arguments": ["cmd_prefix", "output", "lib_suffix"]
        }
        
        # 检查顶层键
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置块缺少必需的 '{key}' 部分")
        
        # 检查路径部分
        paths = config["paths"]
        for key in required_keys["paths"]:
            if key not in paths or not isinstance(paths[key], list):
                raise ValueError(f"paths 部分缺少必需的 '{key}' 数组")
        
        # 检查参数部分
        arguments = config["arguments"]
        for key in required_keys["arguments"]:
            if key not in arguments:
                raise ValueError(f"arguments 部分缺少必需的 '{key}' 值")
        
        return True
    
    def process_config(self, config):
        """处理单个配置块"""
        try:
            # 验证配置结构
            self.validate_config(config)
            
            # 提取配置数据
            paths = config["paths"]
            arguments = config["arguments"]
            
            include_paths = paths["include"]
            lib_paths = paths["lib"]
            source_dirs = paths["source_code"]
            cmd_prefix = arguments["cmd_prefix"]
            output = arguments["output"]
            lib_suffix = arguments["lib_suffix"]
            
            print("\n\033[1;35m" + "=" * 60 + "\033[0m")
            print(f"\033[1;35m开始处理配置块\033[0m")
            print(f"  源文件目录: {source_dirs}")
            print(f"  包含路径: {include_paths}")
            print(f"  库路径: {lib_paths}")
            
            # 收集所有源文件
            source_files = self.collect_source_files(source_dirs)
            if not source_files:
                print("\033[1;33m警告: 未找到任何源文件，跳过此配置块\033[0m")
                return False
            
            # 构建编译命令
            compile_cmd = self.build_compile_command(
                source_files, include_paths, lib_paths, 
                cmd_prefix, output, lib_suffix
            )
            
            # 执行编译命令
            print(f"\n\033[1;34m执行命令: \033[0m{compile_cmd}")
            result = subprocess.run(compile_cmd, shell=True)
            
            if result.returncode != 0:
                print("\033[1;31m编译失败\033[0m")
                return False
            
            # 显示成功信息
            print("\n\033[1;32m编译完成!\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[1;31m处理配置块时出错: {str(e)}\033[0m")
            return False
    
    def collect_source_files(self, source_dirs):
        """收集所有源文件(.c, .cpp)"""
        source_files = []
        for source_dir in source_dirs:
            # 扩展环境变量
            expanded_dir = os.path.expandvars(source_dir)
            if not os.path.exists(expanded_dir):
                print(f"\033[1;33m警告: 目录不存在 - {expanded_dir}\033[0m")
                continue
                
            # 递归查找源文件
            for ext in ["*.cpp", "*.c"]:
                source_files.extend(
                    [str(p) for p in Path(expanded_dir).rglob(ext) if p.is_file()]
                )
        return source_files
    
    def build_include_flags(self, include_paths):
        """构建包含路径参数"""
        return "".join([f' -I"{os.path.expandvars(path)}"' for path in include_paths])
    
    def build_lib_paths_flags(self, lib_paths):
        """构建库路径参数"""
        return "".join([f' -L"{os.path.expandvars(path)}"' for path in lib_paths])
    
    def build_lib_suffix_flags(self, lib_suffix):
        """构建库后缀参数"""
        return "".join([f" {suffix}" for suffix in lib_suffix])
    
    def build_compile_command(self, source_files, include_paths, lib_paths, 
                             cmd_prefix, output, lib_suffix):
        """构建完整的编译命令"""
        # 源文件路径部分
        source_files_str = " ".join([f'"{file}"' for file in source_files])
        
        # 构建各部分参数
        include_flags = self.build_include_flags(include_paths)
        lib_paths_flags = self.build_lib_paths_flags(lib_paths)
        lib_suffix_flags = self.build_lib_suffix_flags(lib_suffix)
        
        # 拼接完整命令
        return (f"{cmd_prefix} {source_files_str} "
                f"{output} "
                f"{include_flags} "
                f"{lib_paths_flags} "
                f"{lib_suffix_flags}")
    
    def run(self):
        """运行整个流程"""
        if not self.load_config():
            return False
        
        # 处理所有启用的配置块
        success = True
        for i, config in enumerate(self.configs, 1):
            print(f"\n\033[1;36m处理配置块 {i}/{len(self.configs)}\033[0m")
            if not self.process_config(config):
                success = False
                print(f"\033[1;31m配置块 {i} 处理失败\033[0m")
            else:
                print(f"\033[1;32m配置块 {i} 处理成功\033[0m")
        
        return success


if __name__ == "__main__":
    compiler = CCG()
    if compiler.run():
        exit(0)
    else:
        exit(1)
