# -*- coding: utf-8 -*-
"""添加库文件路径"""
import sys
import os

def add_release_path():
    # 获取当前脚本所在目录（现在已经是根目录了）
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 根目录就是当前目录
    root_directory = current_directory
    release_directory = os.path.join(root_directory, 'Release')
    
    # 平台相关路径
    windows_directory = os.path.join(release_directory, 'windows')
    linux_directory = os.path.join(release_directory, 'linux')
    
    # Python 包的具体搜索路径
    pyi_w_directory = os.path.join(windows_directory, 'xCoreSDK_python')
    pyi_l_directory = os.path.join(linux_directory, 'xCoreSDK_python')
    
    # 添加到 sys.path
    if root_directory not in sys.path:
        sys.path.append(root_directory)
    if release_directory not in sys.path:
        sys.path.append(release_directory)
    
    # 根据系统添加特定路径
    if sys.platform.startswith('win'):
        sys.path.append(windows_directory)
        sys.path.append(pyi_w_directory)
    else:
        sys.path.append(linux_directory)
        sys.path.append(pyi_l_directory)

# 执行路径添加函数
add_release_path()