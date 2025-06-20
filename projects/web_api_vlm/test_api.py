#!/usr/bin/env python3
"""
MinerU VLM Web API 测试脚本

用法:
    python test_api.py --help
    python test_api.py --url http://localhost:8000 --file test.pdf
    python test_api.py --url http://localhost:8000 --file test.pdf --backend vlm-sglang-engine
"""

import argparse
import json
import os
import requests
import sys
import time
from pathlib import Path


def test_health_check(base_url: str):
    """测试健康检查接口"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_root_endpoint(base_url: str):
    """测试根路径接口"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data.get('message', 'N/A')}")
            print(f"   Version: {data.get('version', 'N/A')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False


def test_vlm_parse(base_url: str, file_path: str, backend: str = "vlm-transformers", 
                   server_url: str = None):
    """测试VLM解析接口"""
    print(f"🔍 Testing VLM parse with {backend}...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # 准备请求数据
        data = {
            "backend": backend,
            "return_images": False,  # 避免返回大量数据
            "return_middle_json": False,
            "return_model_output": False,
            "save_files": False
        }
        
        if server_url:
            data["server_url"] = server_url
        
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
            
            print(f"✅ VLM parse successful")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   File name: {result.get('file_name', 'N/A')}")
            print(f"   Backend: {result.get('backend', 'N/A')}")
            print(f"   Pages processed: {result.get('pages_processed', 'N/A')}")
            
            # 显示MD内容的前100个字符
            md_content = result.get('md_content', '')
            if md_content:
                preview = md_content[:100].replace('\n', ' ')
                print(f"   MD content preview: {preview}...")
            
            return True
        else:
            print(f"❌ VLM parse failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ VLM parse error: {e}")
        return False


def test_batch_parse(base_url: str, file_paths: list, backend: str = "vlm-transformers"):
    """测试批量解析接口"""
    print(f"🔍 Testing batch parse with {len(file_paths)} files...")
    
    # 验证文件是否存在
    valid_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid files for batch processing")
        return False
    
    try:
        # 准备文件
        files = []
        for file_path in valid_files:
            files.append(("files", open(file_path, "rb")))
        
        data = {
            "backend": backend,
            "return_images": False,
            "save_files": False
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/batch_vlm_parse",
            files=files,
            data=data,
            timeout=600  # 10分钟超时
        )
        end_time = time.time()
        
        # 关闭文件
        for _, file_obj in files:
            file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            processing_time = end_time - start_time
            
            print(f"✅ Batch parse successful")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Total files: {result.get('total_files', 0)}")
            print(f"   Successful: {result.get('successful', 0)}")
            print(f"   Failed: {result.get('failed', 0)}")
            
            if result.get('failed_files'):
                print("   Failed files:")
                for failed in result['failed_files']:
                    print(f"     - {failed['file']}: {failed['error']}")
            
            return True
        else:
            print(f"❌ Batch parse failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Batch parse error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MinerU VLM Web API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--file", help="PDF file to test with")
    parser.add_argument("--files", nargs="*", help="Multiple files for batch testing")
    parser.add_argument("--backend", default="vlm-transformers",
                       choices=["vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"],
                       help="VLM backend to test")
    parser.add_argument("--server-url", help="SGLang server URL (for sglang-client backend)")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    parser.add_argument("--skip-parse", action="store_true", help="Skip parse testing")
    parser.add_argument("--skip-batch", action="store_true", help="Skip batch testing")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing MinerU VLM Web API at {args.url}")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # 测试健康检查
    if not args.skip_health:
        total_tests += 2
        if test_health_check(args.url):
            success_count += 1
        if test_root_endpoint(args.url):
            success_count += 1
    
    # 测试VLM解析
    if not args.skip_parse and args.file:
        total_tests += 1
        if test_vlm_parse(args.url, args.file, args.backend, args.server_url):
            success_count += 1
    
    # 测试批量解析
    if not args.skip_batch and args.files:
        total_tests += 1
        if test_batch_parse(args.url, args.files, args.backend):
            success_count += 1
    
    # 显示结果
    print("=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 