import json
import os
import subprocess
import tempfile
import threading
import time
import signal
import atexit
import asyncio
import httpx
from io import StringIO
from pathlib import Path
from typing import Tuple, Union
from threading import RLock, Condition
from contextlib import contextmanager

from fastapi import HTTPException, UploadFile
from loguru import logger

# Import MinerU modules for OCR processing
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.data.data_reader_writer import DataWriter, FileBasedDataWriter
from mineru.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from mineru.utils.config_reader import get_bucket_name, get_s3_config
from mineru.utils.enum_class import MakeMode

# Import pipeline modules for OCR text extraction
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf

# VLM相关导入
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze, aio_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make, mk_blocks_to_markdown, merge_para_with_text
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

# 注册自定义模型
import mineru.model.vlm_hf_model  # 注册 HF transformers 模型
import mineru.model.vlm_sglang_model  # 注册 SGLang 模型

# Supported file extensions
pdf_extensions = [".pdf"]
office_extensions = [".ppt", ".pptx", ".doc", ".docx"]
image_extensions = [".png", ".jpg", ".jpeg"]
text_extensions = [".txt", ".md", ".markdown", ".text", ".rst", ".log"]
all_supported_extensions = pdf_extensions + office_extensions + image_extensions + text_extensions

# 读写锁实现类
class ReadWriteLock:
    """读写锁实现，支持多个读者或单个写者"""

    def __init__(self):
        self._read_ready = Condition(RLock())
        self._readers = 0

    @contextmanager
    def read_lock(self):
        """获取读锁的上下文管理器"""
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

        try:
            yield
        finally:
            self._read_ready.acquire()
            try:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()
            finally:
                self._read_ready.release()

    @contextmanager
    def write_lock(self):
        """获取写锁的上下文管理器"""
        self._read_ready.acquire()
        try:
            while self._readers > 0:
                self._read_ready.wait()
            yield
        finally:
            self._read_ready.release()


# 全局变量跟踪SGLang服务器状态
SGLANG_SERVER_PROCESS = None
CURRENT_BACKEND = "sglang-client"  # 默认使用sglang-client
SGLANG_SERVER_URL = "http://localhost:30000"
SGLANG_MODEL_PATH = None

# 监控线程相关变量
MONITOR_THREAD = None
MONITOR_RUNNING = False
MONITOR_INTERVAL = int(os.getenv("SGLANG_MONITOR_INTERVAL", "10"))  # 检查间隔（秒），可通过环境变量配置

# 超时设置
DEFAULT_VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))  # VLM分析默认超时时间（秒）
DEFAULT_OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "120"))  # OCR分析默认超时时间（秒）

# 健康检查重试设置
HEALTH_CHECK_MAX_RETRIES = int(os.getenv("HEALTH_CHECK_MAX_RETRIES", "3"))  # 健康检查最大重试次数
HEALTH_CHECK_RETRY_DELAYS = [1, 2, 3]  # 重试间隔时间（秒），每次递增

# SGLang服务器管理锁
SGLANG_LOCK = ReadWriteLock()  # 读写锁，用于SGLang服务器状态管理
SERVER_STATE_LOCK = threading.RLock()  # 递归锁，用于服务器状态变更
HEALTH_CHECK_LOCK = threading.Lock()  # 普通锁，用于健康检查


# 错误处理辅助函数
def format_error_message(error: Exception, context: str = "") -> str:
    """
    格式化错误信息，提供更有用的错误描述

    Args:
        error: 异常对象
        context: 错误上下文描述

    Returns:
        str: 格式化后的错误信息
    """
    error_msg = str(error) if str(error) else "未知错误"

    # 根据错误类型和内容提供更具体的描述
    if "connection" in error_msg.lower() or "server" in error_msg.lower():
        return f"服务器连接失败: {error_msg}"
    elif "model" in error_msg.lower() and "load" in error_msg.lower():
        return f"模型加载失败: {error_msg}"
    elif "memory" in error_msg.lower() or "cuda" in error_msg.lower():
        return f"内存或GPU资源不足: {error_msg}"
    elif "format" in error_msg.lower() or "decode" in error_msg.lower():
        return f"文件格式错误: {error_msg}"
    elif "timeout" in error_msg.lower():
        return f"处理超时: {error_msg}"
    elif not error_msg or error_msg == "未知错误":
        return f"{context}处理失败: 发生未知错误，请检查日志获取详细信息"
    else:
        return f"{context}: {error_msg}"


# SGLang服务器管理功能
def kill_sglang_server():
    """安全停止SGLang服务器（需要在写锁保护下调用）"""
    global SGLANG_SERVER_PROCESS

    with SERVER_STATE_LOCK:
        if SGLANG_SERVER_PROCESS:
            try:
                logger.info("正在停止SGLang服务器...")
                SGLANG_SERVER_PROCESS.terminate()
                try:
                    SGLANG_SERVER_PROCESS.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("SGLang服务器未在5秒内停止，强制终止")
                    SGLANG_SERVER_PROCESS.kill()
                SGLANG_SERVER_PROCESS = None
                logger.info("SGLang服务器已停止")
            except Exception as e:
                logger.exception("停止SGLang服务器时发生错误")
                logger.error(f"错误详情: {format_error_message(e, '停止SGLang服务器')}")


def check_sglang_server_health(url: str, timeout: int = 5, max_retries: int = None) -> bool:
    """
    检查SGLang服务器健康状态（带重试机制）

    Args:
        url: 服务器URL
        timeout: 单次请求超时时间
        max_retries: 最大重试次数，默认使用全局配置

    Returns:
        bool: 服务器是否健康
    """
    if max_retries is None:
        max_retries = HEALTH_CHECK_MAX_RETRIES

    with HEALTH_CHECK_LOCK:
        for retry in range(max_retries + 1):
            try:
                response = httpx.get(f"{url}/health", timeout=timeout)
                if response.status_code == 200:
                    if retry > 0:
                        logger.info(f"SGLang服务器健康检查在第{retry + 1}次尝试后成功")
                    return True
                else:
                    logger.warning(f"SGLang服务器健康检查返回状态码: {response.status_code}")
            except Exception as e:
                if retry < max_retries:
                    # 计算重试延迟时间，使用配置的延迟数组或递增延迟
                    if retry < len(HEALTH_CHECK_RETRY_DELAYS):
                        delay = HEALTH_CHECK_RETRY_DELAYS[retry]
                    else:
                        delay = min(retry + 1, 5)  # 最大延迟5秒

                    logger.warning(f"SGLang服务器健康检查失败 (第{retry + 1}次): {str(e)}, {delay}秒后重试...")
                    time.sleep(delay)
                else:
                    logger.error(f"SGLang服务器健康检查最终失败 (共{max_retries + 1}次尝试): {str(e)}")

        return False


def _get_server_process_status() -> bool:
    """获取服务器进程状态（内部函数，需要在锁保护下调用）"""
    global SGLANG_SERVER_PROCESS
    return SGLANG_SERVER_PROCESS is not None and SGLANG_SERVER_PROCESS.poll() is None


def check_sglang_server_health_for_monitor(url: str) -> bool:
    """
    专门用于监控线程的健康检查函数，带重试机制

    Args:
        url: 服务器URL

    Returns:
        bool: 服务器是否健康（经过重试后的最终结果）
    """
    # 使用较短的超时时间进行快速检查
    timeout = 3
    max_retries = HEALTH_CHECK_MAX_RETRIES

    logger.debug(f"开始SGLang服务器健康检查，最大重试{max_retries}次")

    for retry in range(max_retries + 1):
        try:
            response = httpx.get(f"{url}/health", timeout=timeout)
            if response.status_code == 200:
                if retry > 0:
                    logger.info(f"SGLang服务器健康检查在第{retry + 1}次尝试后恢复正常")
                return True
            else:
                logger.warning(f"SGLang服务器健康检查返回异常状态码: {response.status_code}")
        except Exception as e:
            if retry < max_retries:
                # 使用递减的重试延迟，快速确认问题
                if retry < len(HEALTH_CHECK_RETRY_DELAYS):
                    delay = HEALTH_CHECK_RETRY_DELAYS[retry]
                else:
                    delay = min(retry + 1, 3)  # 监控线程使用较短延迟

                logger.warning(f"SGLang服务器健康检查失败 (第{retry + 1}/{max_retries + 1}次): {str(e)}, {delay}秒后重试...")
                time.sleep(delay)
            else:
                logger.error(f"SGLang服务器健康检查彻底失败，已重试{max_retries + 1}次，将触发服务器重启")

    return False


def start_sglang_server() -> bool:
    """启动SGLang服务器（使用写锁保护）"""
    global SGLANG_SERVER_PROCESS, SGLANG_MODEL_PATH

    with SGLANG_LOCK.write_lock():
        with SERVER_STATE_LOCK:
            # 检查是否已经有服务器在运行
            if _get_server_process_status():
                logger.info("SGLang服务器已在运行")
                return True

            try:
                # 获取模型路径
                if not SGLANG_MODEL_PATH:
                    SGLANG_MODEL_PATH = auto_download_and_get_model_root_path("/", "vlm")

                logger.info("正在启动SGLang服务器...")
                logger.info(f"模型路径: {SGLANG_MODEL_PATH}")

                # 设置环境变量以限制显存使用
                env = os.environ.copy()
                env.update({
                    "SGLANG_MEM_FRACTION": "0.6",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.6",
                    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                    "OMP_NUM_THREADS": "8",
                    "MKL_NUM_THREADS": "8"
                })

                # 构建启动命令
                cmd = [
                    "python", "-m", "mineru.model.vlm_sglang_model.server",
                    "--model-path", SGLANG_MODEL_PATH,
                    "--host", "0.0.0.0",
                    "--port", "30000",
                    "--context-length", "20480",
                    "--tensor-parallel-size", "1",
                    "--mem-fraction-static", "0.6",
                    "--quantization", "fp8",
                    "--kv-cache-dtype", "fp8_e5m2",
                    "--chunked-prefill-size", "512",
                    "--enable-memory-saver",
                    "--enable-mixed-chunk",
                    "--allow-auto-truncate",
                    "--stream-interval", "2",
                    "--stream-output"
                ]

                # 启动服务器进程
                SGLANG_SERVER_PROCESS = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    bufsize=1,
                    universal_newlines=False,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )

                # 等待服务器启动
                max_wait_time = 120
                wait_interval = 2
                total_waited = 0

                logger.info("等待SGLang服务器启动...")
                while total_waited < max_wait_time:
                    if SGLANG_SERVER_PROCESS.poll() is not None:
                        logger.error(f"SGLang服务器启动失败，退出码: {SGLANG_SERVER_PROCESS.returncode}")
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
                logger.exception("启动SGLang服务器时发生错误")
                logger.error(f"错误详情: {format_error_message(e, '启动SGLang服务器')}")
                kill_sglang_server()
                return False


def sglang_monitor_thread():
    """SGLang服务器监控线程，定期检查服务器状态并自动重启"""
    global MONITOR_RUNNING, CURRENT_BACKEND

    logger.info(f"SGLang监控线程启动，检查间隔: {MONITOR_INTERVAL}秒")
    last_health_log_time = 0
    health_log_interval = 60  # 每60秒记录一次健康状态日志

    while MONITOR_RUNNING:
        try:
            # 只有当前后端是sglang-client时才进行监控
            if CURRENT_BACKEND == "sglang-client":
                current_time = time.time()

                # 使用读锁检查服务器状态
                with SGLANG_LOCK.read_lock():
                    with SERVER_STATE_LOCK:
                        server_running = _get_server_process_status()

                # 检查进程是否还在运行
                if not server_running:
                    logger.warning("检测到SGLang服务器进程已停止，尝试重启...")
                    restart_sglang_server()
                # 检查服务器健康状态（带重试机制）
                elif not check_sglang_server_health_for_monitor(SGLANG_SERVER_URL):
                    logger.warning("SGLang服务器健康检查多次重试后仍然失败，尝试重启...")
                    restart_sglang_server()
                else:
                    # 只在间隔时间内记录一次健康状态，减少日志输出
                    if current_time - last_health_log_time >= health_log_interval:
                        logger.info("SGLang服务器状态正常")
                        last_health_log_time = current_time

            # 等待下次检查
            time.sleep(MONITOR_INTERVAL)

        except Exception as e:
            logger.exception("SGLang监控线程发生错误")
            logger.error(f"监控错误详情: {format_error_message(e, 'SGLang监控')}")
            time.sleep(MONITOR_INTERVAL)  # 出错后也要等待，避免频繁重试

    logger.info("SGLang监控线程已停止")


def restart_sglang_server():
    """重启SGLang服务器（使用写锁保护）"""
    global CURRENT_BACKEND

    with SGLANG_LOCK.write_lock():
        logger.info("正在重启SGLang服务器...")

        # 先停止现有服务器
        kill_sglang_server()

        # 等待一下确保完全停止
        time.sleep(2)

        # 尝试重新启动
        if start_sglang_server():
            logger.info("SGLang服务器重启成功")
            with SERVER_STATE_LOCK:
                CURRENT_BACKEND = "sglang-client"
        else:
            logger.error("SGLang服务器重启失败，切换到transformers后端")
            with SERVER_STATE_LOCK:
                CURRENT_BACKEND = "transformers"


def start_monitor():
    """启动SGLang监控线程"""
    global MONITOR_THREAD, MONITOR_RUNNING

    if MONITOR_THREAD is not None and MONITOR_THREAD.is_alive():
        logger.info("SGLang监控线程已在运行")
        return

    MONITOR_RUNNING = True
    MONITOR_THREAD = threading.Thread(target=sglang_monitor_thread, daemon=True)
    MONITOR_THREAD.start()
    logger.info("SGLang监控线程已启动")


def stop_monitor():
    """停止SGLang监控线程"""
    global MONITOR_RUNNING, MONITOR_THREAD

    if MONITOR_RUNNING:
        logger.info("正在停止SGLang监控线程...")
        MONITOR_RUNNING = False

        if MONITOR_THREAD and MONITOR_THREAD.is_alive():
            MONITOR_THREAD.join(timeout=5)
            if MONITOR_THREAD.is_alive():
                logger.warning("监控线程未能在5秒内停止")
            else:
                logger.info("SGLang监控线程已停止")

        MONITOR_THREAD = None


def set_monitor_interval(interval: int):
    """设置监控间隔"""
    global MONITOR_INTERVAL

    if interval < 5:
        raise ValueError("监控间隔不能小于5秒")
    if interval > 300:
        raise ValueError("监控间隔不能大于300秒")

    old_interval = MONITOR_INTERVAL
    MONITOR_INTERVAL = interval
    logger.info(f"监控间隔已从{old_interval}秒调整为{interval}秒")

    # 如果监控正在运行，需要重启以应用新间隔
    if MONITOR_RUNNING:
        logger.info("重启监控线程以应用新间隔...")
        stop_monitor()
        time.sleep(1)
        start_monitor()


def get_monitor_interval():
    """获取当前监控间隔"""
    return MONITOR_INTERVAL


def set_health_check_retries(max_retries: int, retry_delays: list = None):
    """
    设置健康检查重试参数

    Args:
        max_retries: 最大重试次数
        retry_delays: 重试延迟时间列表（秒）
    """
    global HEALTH_CHECK_MAX_RETRIES, HEALTH_CHECK_RETRY_DELAYS

    if max_retries < 1:
        raise ValueError("最大重试次数不能小于1")
    if max_retries > 10:
        raise ValueError("最大重试次数不能大于10")

    old_retries = HEALTH_CHECK_MAX_RETRIES
    HEALTH_CHECK_MAX_RETRIES = max_retries

    if retry_delays is not None:
        if len(retry_delays) == 0:
            raise ValueError("重试延迟列表不能为空")
        if any(delay < 0.1 or delay > 10 for delay in retry_delays):
            raise ValueError("重试延迟时间必须在0.1-10秒之间")

        old_delays = HEALTH_CHECK_RETRY_DELAYS.copy()
        HEALTH_CHECK_RETRY_DELAYS = retry_delays.copy()
        logger.info(f"健康检查重试参数已更新: 最大重试次数 {old_retries} -> {max_retries}, 延迟时间 {old_delays} -> {retry_delays}")
    else:
        logger.info(f"健康检查最大重试次数已更新: {old_retries} -> {max_retries}")


def get_health_check_config():
    """获取当前健康检查配置"""
    return {
        "max_retries": HEALTH_CHECK_MAX_RETRIES,
        "retry_delays": HEALTH_CHECK_RETRY_DELAYS.copy()
    }


def initialize_backend():
    """初始化后端，尝试启动SGLang服务器，失败则回退到transformers"""
    global CURRENT_BACKEND

    logger.info("初始化VLM后端...")

    # 首先尝试启动SGLang服务器
    if start_sglang_server():
        with SERVER_STATE_LOCK:
            CURRENT_BACKEND = "sglang-client"
        logger.info("后端初始化完成：使用sglang-client")
        # 启动监控线程
        start_monitor()
    else:
        with SERVER_STATE_LOCK:
            CURRENT_BACKEND = "transformers"
        logger.warning("SGLang服务器启动失败，回退到transformers后端")

    return CURRENT_BACKEND


def cleanup_resources():
    """清理所有资源（使用写锁保护）"""
    with SGLANG_LOCK.write_lock():
        logger.info("正在清理资源...")
        stop_monitor()
        kill_sglang_server()


def signal_handler(signum, frame):
    """处理信号"""
    logger.info(f"接收到信号 {signum}，正在清理...")
    cleanup_resources()
    exit(0)


class MemoryDataWriter(DataWriter):
    """内存数据写入器，用于在内存中暂存数据"""

    def __init__(self):
        self.buffer = StringIO()
        self.binary_files = {}  # 存储二进制文件的字典

    def write(self, filename: str, content):
        """实现抽象方法write"""
        if isinstance(content, str):
            self.write_string(filename, content)
        elif isinstance(content, bytes):
            self.write_binary(filename, content)
        elif isinstance(content, list):
            self.write_jsonl(filename, content)
        else:
            # 尝试转换为字符串
            self.write_string(filename, str(content))

    def write_string(self, filename: str, content: str):
        """写入字符串内容"""
        self.buffer.write(f"=== {filename} ===\n")
        self.buffer.write(content)
        self.buffer.write("\n\n")

    def write_jsonl(self, filename: str, content: list):
        """写入JSONL格式内容"""
        self.buffer.write(f"=== {filename} ===\n")
        for item in content:
            self.buffer.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.buffer.write("\n")

    def write_binary(self, filename: str, content: bytes):
        """写入二进制内容"""
        self.binary_files[filename] = content

    def get_value(self) -> str:
        """获取所有文本内容"""
        return self.buffer.getvalue()

    def get_binary(self, filename: str) -> bytes:
        """获取指定的二进制文件内容"""
        return self.binary_files.get(filename, b"")

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
    Union[S3DataWriter, FileBasedDataWriter, MemoryDataWriter],
    Union[S3DataWriter, FileBasedDataWriter, MemoryDataWriter],
    bytes,
    str
]:
    """
    初始化数据写入器并获取文件内容
    
    Args:
        file_path: 文件路径（本地或S3）
        file: 上传的文件对象
        output_path: 输出路径
        output_image_path: 图像输出路径
        
    Returns:
        tuple: (writer, image_writer, file_bytes, file_extension)
    """
    # 获取文件内容和扩展名
    if file is not None:
        # 处理上传的文件
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    elif file_path is not None:
        # 处理文件路径
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_path.startswith("s3://"):
            # S3文件读取
            bucket = get_bucket_name(file_path)
            ak, sk, endpoint = get_s3_config(bucket)
            temp_reader = S3DataReader(
                "", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
            )
            file_bytes = temp_reader.read(file_path)
        else:
            # 本地文件读取
            with open(file_path, "rb") as f:
                file_bytes = f.read()
    else:
        raise ValueError("必须提供file或file_path中的一个")
    
    # 初始化写入器
    if output_path and output_path.startswith("s3://"):
        # S3写入器
        bucket = get_bucket_name(output_path)
        ak, sk, endpoint = get_s3_config(bucket)
        writer = S3DataWriter(
            output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint
        )
        
        if output_image_path and output_image_path.startswith("s3://"):
            image_bucket = get_bucket_name(output_image_path)
            image_ak, image_sk, image_endpoint = get_s3_config(image_bucket)
            image_writer = S3DataWriter(
                output_image_path, bucket=image_bucket, ak=image_ak, sk=image_sk, endpoint_url=image_endpoint
            )
        else:
            image_writer = MemoryDataWriter()
    elif output_path:
        # 本地文件写入器
        writer = FileBasedDataWriter(output_path)
        if output_image_path:
            image_writer = FileBasedDataWriter(output_image_path)
        else:
            image_writer = MemoryDataWriter()
    else:
        # 内存写入器
        writer = MemoryDataWriter()
        image_writer = MemoryDataWriter()

    return writer, image_writer, file_bytes, file_extension


def detect_file_type(file_extension: str) -> str:
    """
    检测文件类型

    Args:
        file_extension: 文件扩展名

    Returns:
        str: 文件类型 (pdf, office, image, text)
    """
    if file_extension in pdf_extensions:
        return "pdf"
    elif file_extension in office_extensions:
        return "office"
    elif file_extension in image_extensions:
        return "image"
    elif file_extension in text_extensions:
        return "text"
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}")


def process_office_file(file_bytes: bytes, file_extension: str) -> bytes:
    """
    处理Office文件，转换为可处理的格式

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名

    Returns:
        bytes: 处理后的文件字节内容
    """
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"temp{file_extension}")

    try:
        # 写入临时文件
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)

        # 这里可以添加Office文件的具体处理逻辑
        # 目前直接返回原始字节内容
        return file_bytes

    finally:
        # 清理临时文件
        try:
            os.remove(temp_file_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


def process_text_file(file_bytes: bytes, file_extension: str) -> list:
    """
    处理文本文件，直接返回原始文本内容

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名

    Returns:
        list: 包含单页文本内容的列表，类似其他处理函数的返回格式
    """
    try:
        # 尝试多种编码方式解码文本
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
        text_content = None

        for encoding in encodings:
            try:
                text_content = file_bytes.decode(encoding)
                logger.info(f"成功使用 {encoding} 编码解码文本文件")
                break
            except UnicodeDecodeError:
                continue

        if text_content is None:
            # 如果所有编码都失败，使用utf-8并忽略错误
            text_content = file_bytes.decode('utf-8', errors='ignore')
            logger.warning("使用utf-8编码并忽略错误解码文本文件")

        # 对于文本文件，我们将整个内容作为单页返回
        # 这样保持与其他处理函数相同的返回格式（列表）
        return [text_content]

    except Exception as e:
        logger.exception(f"处理文本文件失败: {e}")
        raise ValueError(f"无法处理文本文件: {str(e)}")


async def process_ocr_file(
    file_bytes: bytes,
    file_extension: str,
    lang: str = "ch",
    timeout: int = None
) -> list:
    """
    使用Pipeline进行OCR文本提取，返回分页内容列表

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名
        lang: 语言设置
        timeout: 超时时间（秒），默认使用DEFAULT_OCR_TIMEOUT

    Returns:
        list: 每页的markdown内容列表，类似vlm_parse的md_content_pages
    """
    # 检查文件格式支持
    if file_extension not in all_supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_extension}，支持的格式: {', '.join(all_supported_extensions)}"
        )
    
    # 检测文件类型
    file_type = detect_file_type(file_extension)

    # 对于文本文件，直接返回文本内容，不需要OCR处理
    if file_type == "text":
        logger.info(f"处理文本文件: {file_extension}")
        return process_text_file(file_bytes, file_extension)

    # 根据文件类型进行预处理
    if file_type == "pdf":
        # PDF文件格式转换
        file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)

        # 自动判断是否需要OCR
        parse_method = "ocr" if classify(file_bytes) == "ocr" else "auto"
    elif file_type == "office":
        # Office文件需要OCR处理
        file_bytes = process_office_file(file_bytes, file_extension)
        parse_method = "ocr"
        logger.info(f"处理Office文件: {file_extension}")
    elif file_type == "image":
        # 图像文件直接OCR
        parse_method = "ocr"
        logger.info(f"处理图像文件: {file_extension}")
    
    # 检查文档复杂度并确定超时时间
    complexity_info = check_document_complexity(file_bytes, file_extension)
    effective_timeout = timeout or complexity_info.get("recommended_timeout", DEFAULT_OCR_TIMEOUT)

    logger.info(f"开始OCR分析，文档复杂度: {complexity_info['complexity']}，预估页数: {complexity_info['estimated_pages']}，超时时间: {effective_timeout}秒")

    # 如果有警告，记录日志
    for warning in complexity_info.get("warnings", []):
        logger.warning(warning)

    try:
        # 使用asyncio.to_thread将同步的pipeline_doc_analyze转为异步，并添加超时控制
        start_time = time.time()

        try:
            # 创建进度监控任务
            async def progress_monitor():
                elapsed = 0
                while elapsed < effective_timeout:
                    await asyncio.sleep(30)  # 每30秒报告一次进度
                    elapsed = time.time() - start_time
                    if elapsed < effective_timeout:
                        remaining = effective_timeout - elapsed
                        logger.info(f"OCR分析进行中... 已用时: {elapsed:.1f}秒, 剩余时间: {remaining:.1f}秒")

            # 启动进度监控任务
            progress_task = asyncio.create_task(progress_monitor())

            try:
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = await asyncio.wait_for(
                    asyncio.to_thread(
                        pipeline_doc_analyze,
                        [file_bytes],
                        [lang],
                        parse_method=parse_method,
                        formula_enable=True,  # OCR模式下需要公式识别
                        table_enable=True     # OCR模式下需要表格处理
                    ),
                    timeout=effective_timeout
                )

                # 取消进度监控任务
                progress_task.cancel()

                elapsed_time = time.time() - start_time
                logger.info(f"OCR分析完成，用时: {elapsed_time:.1f}秒")

            except asyncio.TimeoutError:
                progress_task.cancel()
                elapsed_time = time.time() - start_time
                logger.error(f"OCR分析超时（{effective_timeout}秒），实际用时: {elapsed_time:.1f}秒")
                raise HTTPException(
                    status_code=408,
                    detail=f"OCR分析超时（{effective_timeout}秒），请尝试减少文档页数或稍后重试"
                )

        except Exception:
            if 'progress_task' in locals():
                progress_task.cancel()
            logger.exception("OCR分析失败")
            raise
        
        if not infer_results or len(infer_results) == 0:
            raise HTTPException(status_code=500, detail="文档分析失败，无法提取内容")
        
        # 创建临时的图像写入器
        temp_image_writer = MemoryDataWriter()
        
        # 将模型结果转换为中间JSON
        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0] if all_pdf_docs else None
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]
        
        # 转换为中间JSON格式
        if pdf_doc:
            middle_json = pipeline_result_to_middle_json(
                model_list, 
                images_list, 
                pdf_doc, 
                temp_image_writer, 
                _lang, 
                _ocr_enable, 
                formula_enabled=False
            )
        else:
            # 对于非PDF文件，创建一个简化的middle_json
            middle_json = {"pdf_info": []}
            for page_idx, page_model in enumerate(model_list):
                # 这里需要根据实际的数据结构来处理
                # 暂时创建一个简化的页面信息
                page_info = {
                    "page_idx": page_idx,
                    "para_blocks": page_model if isinstance(page_model, list) else []
                }
                middle_json["pdf_info"].append(page_info)
        
        # 从middle_json中提取文本，使用类似vlm_parse的逻辑
        pdf_info = middle_json["pdf_info"]

        # 使用pipeline的make_blocks_to_markdown生成分页内容，类似vlm_parse的逻辑
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown

        # 生成分页内容列表，类似vlm_union_make_paginated的逻辑
        md_content_pages = []
        for page_info in pdf_info:
            paras_of_layout = page_info.get('para_blocks')
            if not paras_of_layout:
                md_content_pages.append("")
                continue

            # 使用pipeline的make_blocks_to_markdown处理每页
            page_markdown = make_blocks_to_markdown(paras_of_layout, MakeMode.MM_MD, "images")
            page_content = '\n\n'.join(page_markdown) if page_markdown else ""
            md_content_pages.append(page_content)

        # 清理临时写入器
        temp_image_writer.close()

        logger.info(f"成功提取{file_type}文件文本，文件类型: {file_extension}，共{len(md_content_pages)}页")
        return md_content_pages
        
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录详细的错误信息和堆栈跟踪
        logger.exception(f"OCR处理{file_type}文件失败")

        # 使用辅助函数格式化错误信息
        detail = format_error_message(e, f"OCR处理{file_type}文件")
        raise HTTPException(status_code=500, detail=detail)


async def process_vlm_file_with_retry(
    file_bytes: bytes,
    file_extension: str,
    image_writer,
    backend: str = None,
    timeout: int = None,
    max_retries: int = 3
) -> Tuple[dict, list, str]:
    """
    带重试机制的VLM文件处理函数

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名
        image_writer: 图像写入器
        backend: VLM后端类型
        timeout: 超时时间（秒）
        max_retries: 最大重试次数（默认3次）

    Returns:
        tuple: (middle_json, infer_result, md_content)
    """
    last_error = None

    for retry in range(max_retries + 1):
        try:
            if retry > 0:
                logger.info(f"第{retry}次重试VLM分析...")
                # 如果是SGLang服务器问题，尝试重启
                if backend == "sglang-client":
                    restart_sglang_server()

            # 调用原始处理函数
            return await process_vlm_file(
                file_bytes=file_bytes,
                file_extension=file_extension,
                image_writer=image_writer,
                backend=backend,
                timeout=timeout
            )

        except HTTPException as e:
            last_error = e
            # 如果是超时错误，增加超时时间再试
            if e.status_code == 408 and retry < max_retries:
                new_timeout = timeout * 1.5 if timeout else DEFAULT_VLM_TIMEOUT * 1.5
                timeout = min(int(new_timeout), 1800)  # 最大30分钟
                logger.warning(f"VLM分析超时，增加超时时间到{timeout}秒并重试")
                continue
            # 如果是服务器错误，尝试切换后端
            elif e.status_code >= 500 and retry < max_retries:
                old_backend = backend
                backend = "transformers" if backend == "sglang-client" else "sglang-client"
                logger.warning(f"VLM服务器错误，从{old_backend}切换到{backend}后端并重试")
                # 如果切换到sglang-client，确保服务器正在运行
                if backend == "sglang-client":
                    with SGLANG_LOCK.read_lock():
                        with SERVER_STATE_LOCK:
                            if not _get_server_process_status():
                                logger.info("切换到sglang-client后端，但服务器未运行，尝试启动...")
                                restart_sglang_server()
                continue
            raise
        except Exception as e:
            last_error = e
            if retry < max_retries:
                logger.exception(f"VLM分析失败，将在3秒后重试: {str(e)}")
                await asyncio.sleep(3)
                continue
            raise

    # 如果所有重试都失败，抛出最后一个错误
    if last_error:
        if isinstance(last_error, HTTPException):
            raise last_error
        raise HTTPException(
            status_code=500,
            detail=format_error_message(last_error, "VLM处理失败，已重试多次")
        )


async def process_vlm_file(
    file_bytes: bytes,
    file_extension: str,
    image_writer,
    backend: str = None,
    timeout: int = None
) -> Tuple[dict, list, str]:
    """
    使用VLM模型处理文件，返回middle_json、infer_result和markdown内容

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名
        image_writer: 图像写入器
        backend: VLM后端类型
        timeout: 超时时间（秒），默认使用DEFAULT_VLM_TIMEOUT

    Returns:
        tuple: (middle_json, infer_result, md_content)
    """
    # 检查文件格式支持
    if file_extension not in all_supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_extension}，支持的格式: {', '.join(all_supported_extensions)}"
        )

    # 确定文件类型
    file_type = detect_file_type(file_extension)

    # 对于文本文件，直接返回文本内容，不需要VLM处理
    if file_type == "text":
        logger.info(f"处理文本文件: {file_extension}")
        md_content_pages = process_text_file(file_bytes, file_extension)
        # 将分页内容合并为单个字符串
        md_content = '\n\n'.join(md_content_pages) if md_content_pages else ""

        # 创建简化的middle_json结构
        middle_json = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "para_blocks": []
                }
            ]
        }

        return middle_json, [], md_content

    # 检查文档复杂度
    complexity_info = check_document_complexity(file_bytes, file_extension)
    logger.info(f"文档复杂度分析: {complexity_info}")

    # 如果有警告，记录日志
    for warning in complexity_info.get("warnings", []):
        logger.warning(warning)

    # 确定使用的后端和超时时间
    effective_backend = backend or CURRENT_BACKEND
    effective_timeout = timeout or complexity_info.get("recommended_timeout", DEFAULT_VLM_TIMEOUT)
    logger.info(f"处理{file_type}文件，使用后端: {effective_backend}，超时时间: {effective_timeout}秒，预估页数: {complexity_info.get('estimated_pages', 1)}")

    try:
        # 根据后端类型准备参数
        vlm_kwargs = {}
        if effective_backend == "sglang-client":
            vlm_kwargs["server_url"] = SGLANG_SERVER_URL
        elif effective_backend == "transformers":
            vlm_kwargs["model_path"] = SGLANG_MODEL_PATH or auto_download_and_get_model_root_path("/", "vlm")

        # 使用异步VLM模型分析文档，添加超时控制
        logger.info(f"开始VLM分析，后端: {effective_backend}, 参数: {vlm_kwargs}")
        start_time = time.time()

        try:
            # 创建一个任务来监控进度
            async def progress_monitor():
                elapsed = 0
                while elapsed < effective_timeout:
                    await asyncio.sleep(30)  # 每30秒报告一次进度
                    elapsed = time.time() - start_time
                    if elapsed < effective_timeout:
                        remaining = effective_timeout - elapsed
                        logger.info(f"VLM分析进行中... 已用时: {elapsed:.1f}秒, 剩余时间: {remaining:.1f}秒")

            # 启动进度监控任务
            progress_task = asyncio.create_task(progress_monitor())

            try:
                # 使用asyncio.wait_for添加超时控制
                middle_json = await asyncio.wait_for(
                    aio_doc_analyze(
                        file_bytes,
                        image_writer=image_writer,
                        backend=effective_backend,
                        **vlm_kwargs
                    ),
                    timeout=effective_timeout
                )

                # 取消进度监控任务
                progress_task.cancel()

                elapsed_time = time.time() - start_time
                logger.info(f"VLM分析完成，用时: {elapsed_time:.1f}秒，开始生成markdown内容")

            except asyncio.TimeoutError:
                progress_task.cancel()
                elapsed_time = time.time() - start_time
                logger.error(f"VLM分析超时（{effective_timeout}秒），实际用时: {elapsed_time:.1f}秒")
                raise HTTPException(
                    status_code=408,
                    detail=f"VLM分析超时（{effective_timeout}秒），请尝试减少文档页数或稍后重试"
                )

        except Exception as vlm_error:
            if 'progress_task' in locals():
                progress_task.cancel()
            logger.exception("VLM分析失败")
            logger.error(f"VLM分析错误详情: {format_error_message(vlm_error, 'VLM分析')}")
            raise

        # 由于异步版本没有返回infer_result，我们设置为空列表
        infer_result = []

        # 生成Markdown内容
        try:
            pdf_info = middle_json["pdf_info"]
            logger.info(f"开始生成markdown内容，pdf_info包含{len(pdf_info)}页")
            md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, "images")
            logger.info("markdown内容生成完成")
        except KeyError as key_error:
            logger.exception("middle_json缺少必要字段")
            logger.error(f"缺少字段详情: {key_error}")
            raise ValueError(f"VLM分析结果格式错误，缺少字段: {key_error}")
        except Exception as make_error:
            logger.exception("生成markdown内容失败")
            logger.error(f"生成markdown错误详情: {format_error_message(make_error, '生成markdown内容')}")
            raise

        logger.info(f"成功处理{file_type}文件，文件类型: {file_extension}，使用后端: {effective_backend}")
        return middle_json, infer_result, md_content

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录详细的错误信息和堆栈跟踪
        logger.exception(f"VLM处理{file_type}文件失败")

        # 使用辅助函数格式化错误信息
        detail = format_error_message(e, f"VLM处理{file_type}文件")
        raise HTTPException(status_code=500, detail=detail)


def get_current_backend():
    """获取当前VLM后端（使用读锁保护）"""
    with SERVER_STATE_LOCK:
        return CURRENT_BACKEND


def get_sglang_server_status():
    """获取SGLang服务器状态（使用读锁保护）"""
    with SGLANG_LOCK.read_lock():
        with SERVER_STATE_LOCK:
            sglang_status = "running" if _get_server_process_status() else "stopped"
            sglang_health = check_sglang_server_health(SGLANG_SERVER_URL) if sglang_status == "running" else False
            monitor_status = "running" if MONITOR_RUNNING and MONITOR_THREAD and MONITOR_THREAD.is_alive() else "stopped"

            return {
                "status": sglang_status,
                "health": sglang_health,
                "url": SGLANG_SERVER_URL,
                "monitor": {
                    "status": monitor_status,
                    "interval": MONITOR_INTERVAL
                }
            }


def validate_backend(backend: str) -> str:
    """验证并规范化VLM后端类型（使用读锁保护）"""
    valid_backends = ["sglang-client", "transformers"]

    with SERVER_STATE_LOCK:
        default_backend = CURRENT_BACKEND

    if backend not in valid_backends:
        logger.warning(f"无效的后端类型: {backend}，使用默认后端: {default_backend}")
        return default_backend

    return backend


def estimate_processing_time(file_bytes: bytes, file_extension: str) -> int:
    """
    估算文档处理时间，用于设置合理的超时时间

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名

    Returns:
        int: 估算的处理时间（秒）
    """
    try:
        file_size_mb = len(file_bytes) / (1024 * 1024)

        # 根据文件类型和大小估算处理时间
        if file_extension in pdf_extensions:
            # PDF文件：每MB大约需要10-15秒
            estimated_time = max(30, int(file_size_mb * 12))
        elif file_extension in image_extensions:
            # 图像文件：每MB大约需要5-8秒
            estimated_time = max(15, int(file_size_mb * 6))
        elif file_extension in office_extensions:
            # Office文件：每MB大约需要15-20秒
            estimated_time = max(45, int(file_size_mb * 18))
        else:
            # 其他文件类型
            estimated_time = max(20, int(file_size_mb * 10))

        # 设置最大超时时间为10分钟
        return min(estimated_time, 600)

    except Exception:
        # 如果估算失败，返回默认值
        return DEFAULT_VLM_TIMEOUT


def check_document_complexity(file_bytes: bytes, file_extension: str) -> dict:
    """
    检查文档复杂度，提供处理建议

    Args:
        file_bytes: 文件字节内容
        file_extension: 文件扩展名

    Returns:
        dict: 包含复杂度信息和建议的字典
    """
    try:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        complexity_info = {
            "file_size_mb": round(file_size_mb, 2),
            "complexity": "low",
            "estimated_pages": 1,
            "recommended_timeout": DEFAULT_VLM_TIMEOUT,
            "warnings": []
        }

        # 根据文件大小判断复杂度
        if file_size_mb > 50:
            complexity_info["complexity"] = "very_high"
            complexity_info["warnings"].append("文件过大，建议分割后处理")
        elif file_size_mb > 20:
            complexity_info["complexity"] = "high"
            complexity_info["warnings"].append("文件较大，处理时间可能较长")
        elif file_size_mb > 5:
            complexity_info["complexity"] = "medium"

        # 估算页数（粗略估算）
        if file_extension in pdf_extensions:
            # PDF平均每页约0.5-2MB
            complexity_info["estimated_pages"] = max(1, int(file_size_mb / 1.0))
        elif file_extension in image_extensions:
            complexity_info["estimated_pages"] = 1
        elif file_extension in office_extensions:
            # Office文件平均每页约0.2-1MB
            complexity_info["estimated_pages"] = max(1, int(file_size_mb / 0.5))

        # 设置推荐超时时间
        complexity_info["recommended_timeout"] = estimate_processing_time(file_bytes, file_extension)

        # 添加页数相关警告
        if complexity_info["estimated_pages"] > 50:
            complexity_info["warnings"].append("文档页数过多，建议分批处理")
        elif complexity_info["estimated_pages"] > 20:
            complexity_info["warnings"].append("文档页数较多，处理时间可能较长")

        return complexity_info

    except Exception as e:
        logger.warning(f"检查文档复杂度失败: {e}")
        return {
            "file_size_mb": 0,
            "complexity": "unknown",
            "estimated_pages": 1,
            "recommended_timeout": DEFAULT_VLM_TIMEOUT,
            "warnings": ["无法分析文档复杂度"]
        }


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

        if not paras_of_layout:
            page_md_list.append("")
            continue

        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
            page_content = '\n\n'.join(page_markdown) if page_markdown else ""
            page_md_list.append(page_content)

    return page_md_list


async def ocr_extract(
    file_path: str = None,
    file: UploadFile = None,
    output_path: str = None,
    output_image_path: str = None,
    lang: str = "ch",
    timeout: int = None
) -> str:
    """
    OCR文本提取函数，类似vlm_parse的调用模式和实现风格

    Args:
        file_path: 文件路径（本地或S3）
        file: 上传的文件对象
        output_path: 输出路径
        output_image_path: 图像输出路径
        lang: 语言设置
        timeout: 超时时间（秒）

    Returns:
        str: 合并后的markdown内容
    """
    try:
        # 初始化写入器并获取文件内容
        writer, image_writer, file_bytes, file_extension = init_writers(
            file_path=file_path,
            file=file,
            output_path=output_path,
            output_image_path=output_image_path
        )

        try:
            # 使用OCR处理文件，获取分页内容
            md_content_pages = await process_ocr_file(
                file_bytes=file_bytes,
                file_extension=file_extension,
                lang=lang,
                timeout=timeout
            )

            # 将分页内容合并为单个字符串，类似vlm_parse的逻辑
            md_content = '\n\n'.join(md_content_pages) if md_content_pages else ""

            # 如果有输出路径，写入结果
            if output_path:
                writer.write("content.md", md_content)
                logger.info(f"OCR结果已写入: {output_path}")

            logger.info(f"OCR提取完成，共处理{len(md_content_pages)}页，内容长度: {len(md_content)}")
            return md_content

        finally:
            # 清理资源
            try:
                writer.close() if hasattr(writer, 'close') else None
                image_writer.close() if hasattr(image_writer, 'close') else None
            except Exception:
                pass

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录详细的错误信息
        logger.exception("OCR提取失败")
        detail = format_error_message(e, "OCR提取")
        raise HTTPException(status_code=500, detail=detail)


# 注册程序退出时的清理函数
atexit.register(cleanup_resources)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

