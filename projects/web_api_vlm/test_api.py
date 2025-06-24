#!/usr/bin/env python3
"""
MinerU VLM Web API 测试脚本

简化版本，直接使用input目录下的文件进行测试，并将MD输出保存到output目录
按照文件名创建独立文件夹，保存MD文件和其他中间输出
"""

import json
import os
import requests
import sys
import time
from pathlib import Path


def test_health_check(base_url: str):
    """测试健康检查接口"""
    print("🔍 正在测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ 健康检查通过")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查错误: {e}")
        return False


def test_root_endpoint(base_url: str):
    """测试根路径接口"""
    print("🔍 正在测试根路径接口...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 根路径接口: {data.get('message', 'N/A')}")
            print(f"   版本: {data.get('version', 'N/A')}")
            return True
        else:
            print(f"❌ 根路径接口失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 根路径接口错误: {e}")
        return False


def ensure_output_dir():
    """确保output目录存在"""
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_additional_outputs(result_data: dict, output_file_dir: Path):
    """保存其他中间输出文件"""
    try:
        # 保存图片文件
        if 'images' in result_data and result_data['images']:
            images_dir = output_file_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            images_data = result_data['images']
            for image_info in images_data:
                if isinstance(image_info, dict) and 'filename' in image_info and 'data' in image_info:
                    image_path = images_dir / image_info['filename']
                    with open(image_path, 'wb') as f:
                        # 假设图片数据是base64编码的
                        import base64
                        f.write(base64.b64decode(image_info['data']))
                    print(f"   💾 图片已保存: {image_path}")
        
        # 保存中间JSON文件
        if 'middle_json' in result_data and result_data['middle_json']:
            middle_json_path = output_file_dir / "middle.json"
            with open(middle_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data['middle_json'], f, ensure_ascii=False, indent=2)
            print(f"   💾 中间JSON已保存: {middle_json_path}")
        
        # 保存模型输出
        if 'model_output' in result_data and result_data['model_output']:
            model_output_path = output_file_dir / "model_output.json"
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data['model_output'], f, ensure_ascii=False, indent=2)
            print(f"   💾 模型输出已保存: {model_output_path}")
            
    except Exception as e:
        print(f"   ⚠️ 保存其他输出文件失败: {e}")


def test_vlm_parse(base_url: str, file_path: str, backend: str = "vlm-transformers"):
    """测试VLM解析接口"""
    print(f"🔍 正在测试VLM解析 ({backend})...")
    print(f"   文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 准备请求数据 - 启用所有输出以获取完整结果
        data = {
            "backend": backend,
            "return_images": True,  # 启用图片返回
            "return_middle_json": True,  # 启用中间JSON返回
            "return_model_output": True,  # 启用模型输出返回
            "save_files": False
        }
        
        # 上传文件
        with open(file_path, "rb") as f:
            files = {"file": f}
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/vlm_parse",
                files=files,
                data=data,
                timeout=300  # 5分钟超时
            )
            end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            processing_time = end_time - start_time
            
            print(f"✅ VLM解析成功")
            print(f"   处理时间: {processing_time:.2f}秒")
            print(f"   文件名: {result.get('file_name', 'N/A')}")
            print(f"   后端: {result.get('backend', 'N/A')}")
            print(f"   处理页数: {result.get('pages_processed', 'N/A')}")
            
            # 创建输出文件夹结构
            try:
                output_dir = ensure_output_dir()
                input_file = Path(file_path)
                output_file_dir = output_dir / input_file.stem
                output_file_dir.mkdir(exist_ok=True)
                
                # 保存MD内容
                md_content = result.get('md_content', '')
                if md_content:
                    # 显示MD内容的前100个字符
                    preview = md_content[:100].replace('\n', ' ')
                    print(f"   MD内容预览: {preview}...")
                    
                    # 保存MD文件到专用文件夹
                    md_filename = f"{input_file.stem}.md"
                    md_path = output_file_dir / md_filename
                    
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    
                    print(f"   💾 MD文件已保存: {md_path}")
                else:
                    print("   ⚠️ 没有返回MD内容")
                
                # 保存其他中间输出
                save_additional_outputs(result, output_file_dir)
                
                print(f"   📁 所有输出文件保存在: {output_file_dir}")
                
            except Exception as e:
                print(f"   ⚠️ 保存文件失败: {e}")
            
            return True
        else:
            print(f"❌ VLM解析失败: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   错误: {error_data.get('error', '未知错误')}")
            except:
                print(f"   原始响应: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ VLM解析错误: {e}")
        return False


def get_input_files():
    """获取input目录下的所有测试文件"""
    # 获取脚本所在目录，然后找input目录
    script_dir = Path(__file__).parent
    input_dir = script_dir / "input"
    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
    
    files = []
    if input_dir.exists():
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(str(file_path))
    
    return files


def main():
    # 固定配置
    base_url = "http://117.50.252.245:8000/"
    backend = "vlm-transformers"
    
    print(f"🚀 正在测试 MinerU VLM Web API")
    print(f"   API地址: {base_url}")
    print(f"   使用后端: {backend}")
    
    # 确保output目录存在
    output_dir = ensure_output_dir()
    print(f"   输出目录: {output_dir}")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # 测试健康检查
    total_tests += 1
    if test_health_check(base_url):
        success_count += 1
    
    # 测试根路径接口
    total_tests += 1
    if test_root_endpoint(base_url):
        success_count += 1
    
    # 获取输入文件
    input_files = get_input_files()
    if not input_files:
        print("⚠️ input目录中没有找到支持的文件 (.pdf, .png, .jpg, .jpeg)")
        print("   请确保input目录存在并包含测试文件")
    else:
        print(f"📁 找到 {len(input_files)} 个测试文件:")
        for file_path in input_files:
            print(f"   - {file_path}")
        print()
        
        # 测试每个文件
        for file_path in input_files:
            total_tests += 1
            if test_vlm_parse(base_url, file_path, backend):
                success_count += 1
            print()  # 空行分隔
    
    # 显示结果
    print("=" * 50)
    print(f"📊 测试结果: {success_count}/{total_tests} 个测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试都通过了!")
        print(f"📁 输出文件保存在: {output_dir}")
        return 0
    else:
        print("💥 部分测试失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 