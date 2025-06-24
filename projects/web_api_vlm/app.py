import json
import os
import tempfile
import subprocess
import threading
import time
import signal
import atexit
from base64 import b64encode

from io import StringIO
from pathlib import Path
from typing import Tuple, Union

import uvicorn
import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

# Import MinerU VLM modules
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze, aio_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make, mk_blocks_to_markdown
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env
from mineru.data.data_reader_writer import DataWriter, FileBasedDataWriter
from mineru.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from mineru.utils.config_reader import get_bucket_name, get_s3_config
from mineru.utils.enum_class import MakeMode
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

# **重要：添加这两行导入来注册自定义模型**
import mineru.model.vlm_hf_model  # 注册 HF transformers 模型
import mineru.model.vlm_sglang_model  # 注册 SGLang 模型

# 全局变量跟踪SGLang服务器状态
SGLANG_SERVER_PROCESS = None
CURRENT_BACKEND = "sglang-client"  # 默认使用sglang-client
SGLANG_SERVER_URL = "http://localhost:30000"
SGLANG_MODEL_PATH = None

app = FastAPI(
    title="MinerU VLM Web API",
    description="基于MinerU 2.0 VLM模型的多格式文件解析API服务",
    version="2.0.0"
)

# Supported file extensions
pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]
all_supported_extensions = pdf_extensions + office_extensions + image_extensions


def kill_sglang_server():
    """安全停止SGLang服务器"""
    global SGLANG_SERVER_PROCESS
    if SGLANG_SERVER_PROCESS:
        try:
            logger.info("正在停止SGLang服务器...")
            SGLANG_SERVER_PROCESS.terminate()
            # 给进程一些时间优雅退出
            try:
                SGLANG_SERVER_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("SGLang服务器未在5秒内停止，强制终止")
                SGLANG_SERVER_PROCESS.kill()
            SGLANG_SERVER_PROCESS = None
            logger.info("SGLang服务器已停止")
        except Exception as e:
            logger.error(f"停止SGLang服务器时发生错误: {e}")


def check_sglang_server_health(url: str, timeout: int = 5) -> bool:
    """检查SGLang服务器健康状态"""
    try:
        response = httpx.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def start_sglang_server() -> bool:
    """启动SGLang服务器"""
    global SGLANG_SERVER_PROCESS, SGLANG_MODEL_PATH
    
    try:
        # 获取模型路径
        if not SGLANG_MODEL_PATH:
            SGLANG_MODEL_PATH = auto_download_and_get_model_root_path("/", "vlm")
        
        logger.info("正在启动SGLang服务器...")
        logger.info(f"模型路径: {SGLANG_MODEL_PATH}")
        
        # 构建启动命令
        cmd = [
            "python", "-m", "mineru.model.vlm_sglang_model.server",
            "--model-path", SGLANG_MODEL_PATH,
            "--host", "0.0.0.0",
            "--port", "30000"
        ]
        
        # 启动服务器进程
        SGLANG_SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None  # 在Unix系统上创建新的进程组
        )
        
        # 等待服务器启动
        max_wait_time = 120  # 最大等待时间（秒）
        wait_interval = 2
        total_waited = 0
        
        logger.info("等待SGLang服务器启动...")
        while total_waited < max_wait_time:
            if SGLANG_SERVER_PROCESS.poll() is not None:
                # 进程已退出
                stdout, stderr = SGLANG_SERVER_PROCESS.communicate()
                logger.error(f"SGLang服务器启动失败:")
                logger.error(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                logger.error(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                SGLANG_SERVER_PROCESS = None
                return False
            
            if check_sglang_server_health(SGLANG_SERVER_URL):
                logger.info("SGLang服务器启动成功！")
                return True
            
            time.sleep(wait_interval)
            total_waited += wait_interval
            logger.info(f"等待SGLang服务器启动... ({total_waited}/{max_wait_time}秒)")
        
        # 超时
        logger.error("SGLang服务器启动超时")
        kill_sglang_server()
        return False
        
    except Exception as e:
        logger.error(f"启动SGLang服务器时发生错误: {e}")
        kill_sglang_server()
        return False


def initialize_backend():
    """初始化后端，尝试启动SGLang服务器，失败则回退到transformers"""
    global CURRENT_BACKEND
    
    logger.info("初始化VLM后端...")
    
    # 首先尝试启动SGLang服务器
    if start_sglang_server():
        CURRENT_BACKEND = "sglang-client"
        logger.info("后端初始化完成：使用sglang-client")
    else:
        CURRENT_BACKEND = "transformers"
        logger.warning("SGLang服务器启动失败，回退到transformers后端")
    
    return CURRENT_BACKEND


# 注册退出处理器
atexit.register(kill_sglang_server)

# 处理信号
def signal_handler(signum, frame):
    logger.info(f"接收到信号 {signum}，正在清理...")
    kill_sglang_server()
    exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class MemoryDataWriter(DataWriter):
    """内存数据写入器，用于在内存中暂存数据"""
    
    def __init__(self):
        self.buffer = StringIO()
        self.binary_files = {}  # 存储二进制文件的字典

    def write(self, path: str, data: bytes) -> None:
        """写入二进制数据"""
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            # 对于二进制数据，不进行UTF-8解码，而是存储在单独的字典中
            # 这样可以避免图像文件等二进制数据的UTF-8解码错误
            import base64
            self.binary_files[path] = base64.b64encode(data).decode('ascii')
            # 在文本buffer中记录文件引用
            self.buffer.write(f"\n[BINARY_FILE_REF: {path}]\n")

    def write_string(self, path: str, data: str) -> None:
        """写入字符串数据"""
        self.buffer.write(data)

    def get_value(self) -> str:
        """获取buffer内容"""
        return self.buffer.getvalue()
    
    def get_binary_files(self) -> dict:
        """获取存储的二进制文件"""
        return self.binary_files

    def close(self):
        """关闭buffer"""
        self.buffer.close()
        self.binary_files.clear()


def init_writers(
    file_path: str = None,
    file: UploadFile = None,
    output_path: str = None,
    output_image_path: str = None,
) -> Tuple[
    Union[S3DataWriter, MemoryDataWriter],
    Union[S3DataWriter, MemoryDataWriter],
    bytes,
    str,
]:
    """
    初始化写入器

    Args:
        file_path: 文件路径（本地路径或S3路径）
        file: 上传的文件对象
        output_path: 输出目录路径
        output_image_path: 图像输出目录路径

    Returns:
        Tuple[writer, image_writer, file_bytes, file_extension]: 返回初始化的写入器和文件内容
    """
    file_extension = None
    
    if file_path:
        # 处理文件路径（本地或S3）
        is_s3_path = file_path.startswith("s3://")
        file_extension = os.path.splitext(file_path)[1]
        
        if is_s3_path:
            bucket = get_bucket_name(file_path)
            ak, sk, endpoint = get_s3_config(bucket)

            writer = S3DataWriter(
                output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            image_writer = S3DataWriter(
                output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            # 临时创建reader读取文件内容
            temp_reader = S3DataReader(
                "", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            file_bytes = temp_reader.read(file_path)
        else:
            # 本地文件路径 - 只读取文件内容，不创建本地写入器
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            # 强制使用内存写入器，避免本地文件存储
            writer = MemoryDataWriter()
            image_writer = MemoryDataWriter()
    else:
        # 处理上传的文件
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ""

        # 检查输出路径是否为S3路径
        is_s3_path = output_path and output_path.startswith("s3://")
        
        if is_s3_path:
            bucket = get_bucket_name(output_path)
            ak, sk, endpoint = get_s3_config(bucket)

            writer = S3DataWriter(
                output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            image_writer = S3DataWriter(
                output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
        else:
            # 不是S3路径，使用内存写入器，避免本地文件存储
            writer = MemoryDataWriter()
            image_writer = MemoryDataWriter()

    return writer, image_writer, file_bytes, file_extension


def detect_file_type(file_extension: str) -> str:
    """
    检测文件类型
    
    Args:
        file_extension: 文件扩展名
        
    Returns:
        文件类型: 'pdf', 'office', 'image'
    """
    if file_extension in pdf_extensions:
        return "pdf"
    elif file_extension in office_extensions:
        return "office"
    elif file_extension in image_extensions:
        return "image"
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}")


def process_office_file(file_bytes: bytes, file_extension: str) -> bytes:
    """
    处理Office文件，转换为可处理的格式
    
    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名
        
    Returns:
        处理后的文件字节内容
    """
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"temp_file{file_extension}")
    
    try:
        # 写入临时文件
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)
        
        # 这里可以添加Office文件的特殊处理逻辑
        # 目前直接返回原始字节内容
        return file_bytes
        
    finally:
        # 清理临时文件
        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except:
            pass


async def process_vlm_file(
    file_bytes: bytes,
    file_extension: str,
    image_writer: Union[S3DataWriter, MemoryDataWriter],
    backend: str = None,
) -> Tuple[dict, list, str]:
    """
    使用VLM模型处理文件（异步版本）
    
    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名
        image_writer: 图像写入器
        backend: VLM后端类型
        
    Returns:
        Tuple[middle_json, infer_result, md_content]: 返回处理结果
    """
    # 检查文件格式支持
    if file_extension not in all_supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_extension}，支持的格式: {', '.join(all_supported_extensions)}"
        )
    
    # 检测文件类型
    file_type = detect_file_type(file_extension)
    
    # 根据文件类型进行预处理
    if file_type == "pdf":
        # PDF文件格式转换
        file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)
    elif file_type == "office":
        # Office文件处理
        file_bytes = process_office_file(file_bytes, file_extension)
        logger.info(f"处理Office文件: {file_extension}")
    elif file_type == "image":
        # 图像文件直接处理
        logger.info(f"处理图像文件: {file_extension}")
    
    # 使用当前可用的后端，如果指定了backend则覆盖
    effective_backend = backend if backend else CURRENT_BACKEND
    
    try:
        # 根据后端类型准备参数
        vlm_kwargs = {}
        if effective_backend == "sglang-client":
            vlm_kwargs["server_url"] = SGLANG_SERVER_URL
        elif effective_backend == "transformers":
            vlm_kwargs["model_path"] = SGLANG_MODEL_PATH or auto_download_and_get_model_root_path("/", "vlm")
        
        # 使用异步VLM模型分析文档 - 注意aio_doc_analyze只返回middle_json
        middle_json = await aio_doc_analyze(
            file_bytes,
            image_writer=image_writer,
            backend=effective_backend,
            **vlm_kwargs
        )
        
        # 由于异步版本没有返回infer_result，我们设置为空列表
        infer_result = []
        
        # 生成Markdown内容
        pdf_info = middle_json["pdf_info"]
        md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, "images")
        
        logger.info(f"成功处理{file_type}文件，文件类型: {file_extension}，使用后端: {effective_backend}")
        return middle_json, infer_result, md_content
        
    except Exception as e:
        logger.error(f"VLM处理{file_type}文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"VLM处理{file_type}文件失败: {str(e)}")


def validate_backend(backend: str) -> str:
    """验证并规范化VLM后端类型"""
    valid_backends = ["sglang-client", "transformers"]  # 只支持这两种后端
    default_backend = CURRENT_BACKEND  # 使用当前可用的后端
    
    if backend not in valid_backends:
        logger.warning(f"无效的后端类型: {backend}，使用默认后端: {default_backend}")
        return default_backend
    
    return backend


def vlm_union_make_paginated(pdf_info_dict: list, make_mode: str, img_buket_path: str = '') -> list:
    """
    生成分页的Markdown内容，每页对应原始PDF的一页
    
    Args:
        pdf_info_dict: PDF页面信息列表
        make_mode: 生成模式
        img_buket_path: 图片路径
        
    Returns:
        list: 每个元素是一页的Markdown内容字符串
    """
    page_md_list = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx')
        
        if not paras_of_layout:
            # 如果当前页面没有内容，添加空字符串
            page_md_list.append("")
            continue
            
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
            # 将当前页面的所有markdown块合并成一个字符串
            page_content = '\n\n'.join(page_markdown) if page_markdown else ""
            page_md_list.append(page_content)
    
    return page_md_list


def encode_image(image_path: str) -> str:
    """将图像文件编码为base64字符串（已弃用，当前版本不使用本地文件存储）"""
    try:
        with open(image_path, "rb") as f:
            return b64encode(f.read()).decode()
    except Exception as e:
        logger.warning(f"图像编码失败 {image_path}: {e}")
        return ""


@app.on_event("startup")
async def startup_event():
    """应用启动事件：初始化后端"""
    initialize_backend()


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "MinerU VLM Web API",
        "version": "2.0.0",
        "description": "基于MinerU 2.0 VLM模型的多格式文件解析API服务",
        "current_backend": CURRENT_BACKEND,
        "supported_backends": ["sglang-client", "transformers"],
        "supported_formats": {
            "pdf": pdf_extensions,
            "office": office_extensions, 
            "image": image_extensions
        },
        "features": {
            "pagination": "支持按原始页面分页输出Markdown内容",
            "storage": "支持内存处理和S3存储",
            "formats": "支持PDF、Office文档和图像文件"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    sglang_status = "running" if SGLANG_SERVER_PROCESS and SGLANG_SERVER_PROCESS.poll() is None else "stopped"
    sglang_health = check_sglang_server_health(SGLANG_SERVER_URL) if sglang_status == "running" else False
    
    return {
        "status": "healthy", 
        "service": "mineru-vlm-api",
        "current_backend": CURRENT_BACKEND,
        "sglang_server": {
            "status": sglang_status,
            "health": sglang_health,
            "url": SGLANG_SERVER_URL
        }
    }


@app.post(
    "/vlm_parse",
    tags=["VLM解析"],
    summary="使用VLM模型解析多格式文件",
    description="使用视觉语言模型(VLM)解析PDF、Office文档或图像文件，支持内存处理和S3存储"
)
async def vlm_parse(
    file: UploadFile = None,
    file_path: str = Form(None, description="文件路径（本地路径或S3路径）"),
    backend: str = Form(None, description="VLM后端类型: sglang-client, transformers"),
    output_dir: str = Form("output", description="输出目录（S3路径）"),
    return_images: bool = Form(False, description="是否返回图像信息"),
    return_middle_json: bool = Form(False, description="是否返回中间JSON格式"),
    is_json_md_dump: bool = Form(False, description="是否保存文件到S3"),
    return_paginated_md: bool = Form(False, description="是否返回分页的Markdown内容数组")
):
    """
    使用VLM模型解析多格式文件
    
    Args:
        file: 上传的文件（PDF、Office文档或图像）
        file_path: 文件路径（本地路径或S3路径）
        backend: VLM后端类型 (sglang-client, transformers)
        output_dir: 输出目录（S3路径，如 s3://bucket/path）
        return_images: 是否返回图像信息
        return_middle_json: 是否返回中间JSON
        is_json_md_dump: 是否保存到S3
        return_paginated_md: 是否返回分页的Markdown内容数组
        
    Returns:
        解析结果，包含md_content等字段。如果return_paginated_md为True，则返回md_content_pages数组
    """
    try:
        # 验证输入参数 - 优先使用file，没有file才使用file_path
        if file is None and file_path is None:
            return JSONResponse(
                content={"error": "必须提供file或file_path中的一个"},
                status_code=400
            )
        
        # 优先使用file参数
        use_uploaded_file = file is not None
        
        # 获取文件名 - 优先使用file
        if use_uploaded_file:
            file_name = os.path.basename(file.filename).split(".")[0] if file.filename else "uploaded_file"
        else:
            file_name = os.path.basename(file_path).split(".")[0]
            
        output_path = f"{output_dir}/{file_name}"
        output_image_path = f"{output_path}/images"

        # 初始化写入器并获取文件内容
        writer, image_writer, file_bytes, file_extension = init_writers(
            file_path=None if use_uploaded_file else file_path,
            file=file if use_uploaded_file else None,
            output_path=output_path,
            output_image_path=output_image_path,
        )
        
        if not file_extension:
            return JSONResponse(
                content={"error": "无法确定文件类型"},
                status_code=400
            )
        
        # 验证并规范化后端类型
        validated_backend = validate_backend(backend) if backend else CURRENT_BACKEND
        
        # 处理文件
        middle_json, infer_result, md_content = await process_vlm_file(
            file_bytes=file_bytes,
            file_extension=file_extension,
            image_writer=image_writer,
            backend=validated_backend
        )
        
        # 使用MemoryDataWriter来获取结果
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()
        
        md_content_writer.write_string("", md_content)
        middle_json_writer.write_string("", json.dumps(middle_json, ensure_ascii=False, indent=2))
        
        # 检测文件类型用于返回信息
        file_type = detect_file_type(file_extension)
        
        # 构建响应数据
        data = {
            "md_content": md_content,  # 始终返回md内容
            "file_name": file_name,
            "file_type": file_type,
            "file_extension": file_extension,
            "backend": validated_backend
        }
        
        # 添加分页Markdown内容（如果请求）
        if return_paginated_md:
            pdf_info = middle_json["pdf_info"]
            paginated_md = vlm_union_make_paginated(pdf_info, MakeMode.MM_MD, "images")
            data["md_content_pages"] = paginated_md
            data["total_pages"] = len(paginated_md)
        
        # 添加可选的返回数据
        if return_middle_json:
            data["middle_json"] = middle_json
        
        if return_images:
            # 返回图像数据
            if isinstance(image_writer, MemoryDataWriter):
                # 内存模式下，返回存储的二进制文件信息
                binary_files = image_writer.get_binary_files()
                data["images"] = {
                    "count": len(binary_files),
                    "files": binary_files,
                    "message": "图像已在内存中处理并编码为base64格式"
                }
            elif output_path.startswith("s3://"):
                # S3存储的图像处理（暂时不实现base64返回，因为需要从S3读取）
                data["images"] = {"message": "S3存储模式下暂不支持返回图像base64编码"}
            else:
                # 其他情况
                data["images"] = {"message": "图像信息已包含在解析结果中"}
        
        # 保存文件（如果请求）
        if is_json_md_dump:
            writer.write_string(f"{file_name}.md", md_content)
            writer.write_string(f"{file_name}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=2))
            
            # 如果有分页内容，也保存分页文件
            if return_paginated_md:
                paginated_md = data["md_content_pages"]
                for i, page_content in enumerate(paginated_md):
                    writer.write_string(f"{file_name}_page_{i+1}.md", page_content)
            
            data["saved_path"] = output_path
        
        # 清理内存写入器
        md_content_writer.close()
        middle_json_writer.close()
        
        logger.info(f"成功处理文件: {file_name} ({file_type})，使用VLM {validated_backend}后端")
        return JSONResponse(content=data, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VLM处理过程中发生意外错误: {e}")
        return JSONResponse(
            content={"error": f"处理文件时发生错误: {str(e)}"},
            status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    ) 