#!/usr/bin/env python3
"""
MinerU 模型下载和检查脚本
用于在启动服务前检查并下载必要的模型文件
"""

import os
import glob
import sys
import subprocess
import traceback
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.enum_class import ModelPath


def check_file_exists(cache_dir, relative_path):
    """检查特定模型文件是否存在"""
    # 检查多种可能的路径格式
    possible_paths = [
        os.path.join(cache_dir, relative_path),
        os.path.join(cache_dir, f'models--OpenDataLab--PDF-Extract-Kit-1.0', 'snapshots', '*', relative_path),
        os.path.join(cache_dir, f'models--opendatalab--PDF-Extract-Kit-1.0', 'snapshots', '*', relative_path),
    ]
    
    for pattern in possible_paths:
        if '*' in pattern:
            matches = glob.glob(pattern)
            if matches and any(os.path.exists(match) for match in matches):
                return True
        else:
            if os.path.exists(pattern):
                return True
    return False


def check_pipeline_models_exist():
    """检查Pipeline模型是否已存在"""
    cache_dir = '/root/.cache'
    
    # Pipeline模型列表
    pipeline_models = [
        ModelPath.doclayout_yolo,           # models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt
        ModelPath.yolo_v8_mfd,             # models/MFD/YOLO/yolo_v8_ft.pt  
        ModelPath.unimernet_small,         # models/MFR/unimernet_hf_small_2503
        ModelPath.pytorch_paddle,          # models/OCR/paddleocr_torch
        ModelPath.layout_reader,           # models/ReadingOrder/layout_reader
        ModelPath.slanet_plus             # models/TabRec/SlanetPlus/slanet-plus.onnx
    ]
    
    missing_models = []
    existing_models = []
    
    for model_path in pipeline_models:
        if check_file_exists(cache_dir, model_path):
            existing_models.append(model_path)
            print(f'✓ Pipeline模型已存在: {model_path}')
        else:
            missing_models.append(model_path)
            print(f'❌ Pipeline模型缺失: {model_path}')
    
    print(f'Pipeline模型统计: {len(existing_models)}个已存在, {len(missing_models)}个缺失')
    return len(missing_models) == 0


def check_vlm_models_exist():
    """检查VLM模型是否已存在"""
    cache_dir = '/root/.cache'
    
    # 检查可能的VLM模型目录模式
    patterns = [
        'models--OpenDataLab--MinerU2.0-2505-0.9B',
        'models--opendatalab--MinerU2.0-2505-0.9B', 
        'OpenDataLab/MinerU2.0-2505-0.9B',
        'opendatalab/MinerU2.0-2505-0.9B'
    ]
    
    for pattern in patterns:
        full_path = os.path.join(cache_dir, pattern)
        if os.path.exists(full_path):
            # 检查关键文件
            key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
            found_files = []
            for key_file in key_files:
                if os.path.exists(os.path.join(full_path, key_file)):
                    found_files.append(key_file)
            
            if found_files:
                print(f'✓ VLM模型已存在: {full_path}')
                print(f'  找到关键文件: {found_files}')
                return True
    
    # 也检查snapshots目录
    snapshot_dirs = glob.glob(f'{cache_dir}/models--*MinerU2.0-2505-0.9B*/snapshots/*/') 
    for snapshot_dir in snapshot_dirs:
        key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        found_files = []
        for key_file in key_files:
            if os.path.exists(os.path.join(snapshot_dir, key_file)):
                found_files.append(key_file)
        
        if found_files:
            print(f'✓ VLM模型已存在: {snapshot_dir}')
            print(f'  找到关键文件: {found_files}')
            return True
    
    print('❌ VLM模型不存在')
    return False


def download_missing_models():
    """下载缺失的模型"""
    pipeline_exists = check_pipeline_models_exist()
    vlm_exists = check_vlm_models_exist()
    
    download_success = True
    
    # 下载Pipeline模型（如果缺失）
    if not pipeline_exists:
        print('\n🔄 开始下载Pipeline模型...')
        try:
            # 下载所有Pipeline模型
            pipeline_model_paths = [
                ModelPath.doclayout_yolo,
                ModelPath.yolo_v8_mfd,
                ModelPath.unimernet_small,
                ModelPath.pytorch_paddle,
                ModelPath.layout_reader,
                ModelPath.slanet_plus
            ]
            
            for model_path in pipeline_model_paths:
                print(f'📥 下载: {model_path}')
                try:
                    result_path = auto_download_and_get_model_root_path(model_path, repo_mode='pipeline')
                    print(f'✅ 完成: {model_path} -> {result_path}')
                except Exception as e:
                    print(f'❌ 失败: {model_path} - {e}')
                    download_success = False
                    
            print('✅ Pipeline模型下载完成')
        except Exception as e:
            print(f'❌ Pipeline模型下载失败: {e}')
            download_success = False
    else:
        print('✅ Pipeline模型检查完成，无需下载')
    
    # 下载VLM模型（如果缺失）
    if not vlm_exists:
        print('\n🔄 开始下载VLM模型...')
        try:
            vlm_path = auto_download_and_get_model_root_path('/', repo_mode='vlm')
            print(f'✅ VLM模型下载完成: {vlm_path}')
            
            # 验证下载结果
            if os.path.exists(vlm_path):
                print('📁 VLM模型文件验证:')
                subprocess.run(['ls', '-la', vlm_path], check=False)
            
        except Exception as e:
            print(f'❌ VLM模型下载失败: {e}')
            download_success = False
    else:
        print('✅ VLM模型检查完成，无需下载')
    
    return download_success


def check_local_models():
    """检查本地模型配置"""
    print("检查本地模型配置...")
    try:
        from mineru.utils.config_reader import get_local_models_dir
        models_config = get_local_models_dir()
        if models_config:
            print('✓ 本地模型配置已找到')
            if 'vlm' in models_config:
                print(f'✓ VLM模型路径: {models_config["vlm"]}')
            if 'pipeline' in models_config:
                print(f'✓ Pipeline模型路径: {models_config["pipeline"]}')
                
            # 检查本地路径是否真实存在
            for model_type, path in models_config.items():
                if path and os.path.exists(path):
                    print(f'✓ {model_type}模型路径存在: {path}')
                    subprocess.run(['ls', '-la', path], check=False)
                elif path:
                    print(f'⚠️  {model_type}模型路径不存在: {path}')
        else:
            print('⚠ 未找到本地模型配置')
    except Exception as e:
        print(f'⚠ 检查模型时出错: {e}')


def show_cache_status():
    """显示缓存目录状态"""
    print('\n📂 当前模型缓存状态:')
    subprocess.run(['du', '-sh', '/root/.cache'], check=False)
    subprocess.run(['find', '/root/.cache', '-name', '*.pt', '-o', '-name', '*.onnx', '-o', '-name', 'config.json'], check=False)


def main():
    """主函数"""
    model_source = os.environ.get('MINERU_MODEL_SOURCE', 'auto')
    
    print('\n📋 开始模型检查和下载流程...')
    
    if model_source != "local":
        # 智能模型下载 - 检查所有模型并在需要时下载
        print("检查并下载网络模型（如果需要）...")
        try:
            success = download_missing_models()
            if success:
                print('\n🎉 所有模型检查和下载完成！')
            else:
                print('\n⚠️  部分模型下载失败，但服务将继续启动')
                print('   模型将在首次使用时自动下载')
                
            show_cache_status()
            
        except Exception as e:
            print(f'❌ 模型处理过程出错: {e}')
            print('继续启动服务，模型将在首次使用时下载...')
            traceback.print_exc()
    else:
        # 检查本地模型是否可用
        check_local_models()
    
    print("模型检查完成！")


if __name__ == "__main__":
    main() 