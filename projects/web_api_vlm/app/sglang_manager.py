"""
SGLang Server Management Module

This module handles SGLang server startup, health monitoring, and automatic recovery.
Referenced from: https://github.com/rstarall/MinerU/blob/master/projects/web_api_vlm/ocr.py
"""

import os
import time
import signal
import subprocess
import threading
import atexit
from typing import Optional, Dict, Any
from pathlib import Path

import httpx
from loguru import logger

# Import to register custom MinerU2 model with transformers before SGLang startup
import mineru.model.vlm_hf_model
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


class SGLangServerManager:
    """SGLang server lifecycle management"""
    
    def __init__(
        self,
        model_path: str = None,
        host: str = "0.0.0.0",
        port: int = 30000,
        tp_size: int = 1,
        mem_fraction: float = 0.65,  # 保守：降到 0.65，为 torch.compile 预留更多内存
        trust_remote_code: bool = True,
        attention_backend: str = "flashinfer",
        chunked_prefill_size: int = 3072,  # 保守：可进一步降到 2048，减少内存占用
        max_running_requests: int = 64,   # 保守：降到 64，减少并发内存需求
        enable_torch_compile: bool = True,  # 🚫 可禁用，避免 autotuning 内存问题
        disable_custom_all_reduce: bool = False,
    ):
        # 获取模型路径 - 使用 MinerU 的模型下载工具
        if model_path is None:
            # 优先使用环境变量，然后使用 MinerU 的模型路径
            model_path = os.getenv("SGLANG_MODEL_PATH")
            if not model_path:
                model_path = auto_download_and_get_model_root_path("/", "vlm")
        
        self.model_path = model_path
        self.host = os.getenv("SGLANG_HOST", host)
        self.port = int(os.getenv("SGLANG_PORT", str(port)))
        self.tp_size = int(os.getenv("SGLANG_TP_SIZE", str(tp_size)))
        self.mem_fraction = float(os.getenv("SGLANG_MEM_FRACTION", str(mem_fraction)))
        self.trust_remote_code = os.getenv("SGLANG_TRUST_REMOTE_CODE", str(trust_remote_code)).lower() == "true"
        self.attention_backend = os.getenv("SGLANG_ATTENTION_BACKEND", attention_backend)
        self.chunked_prefill_size = chunked_prefill_size
        self.max_running_requests = max_running_requests
        self.enable_torch_compile = enable_torch_compile
        self.disable_custom_all_reduce = disable_custom_all_reduce
        
        self.server_url = f"http://{self.host}:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.should_monitor = False
        self._lock = threading.RLock()
        
        # 健康检查重试设置 - 参考原始配置
        self.HEALTH_CHECK_MAX_RETRIES = int(os.getenv("HEALTH_CHECK_MAX_RETRIES", "3"))
        self.HEALTH_CHECK_RETRY_DELAYS = [1, 2, 3]  # 重试间隔时间（秒），每次递增
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down SGLang server...")
        self.cleanup()
    
    def _build_command(self) -> list:
        """Build SGLang server startup command - 使用 MinerU 的 SGLang 服务器"""
        cmd = [
            "python", "-m", "mineru.model.vlm_sglang_model.server",
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tp-size", str(self.tp_size),
            "--mem-fraction", str(self.mem_fraction),
            "--chunked-prefill-size", str(self.chunked_prefill_size),
            "--max-running-requests", str(self.max_running_requests),
        ]
        
        # 🛡️ 保守配置参数（稳定优先，避免内存问题）
        stability_params = [
            "--context-length", "24576", # 适中的上下文长度，减少内存占用
        ]
        cmd.extend(stability_params)
        
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if self.attention_backend:
            cmd.extend(["--attention-backend", self.attention_backend])
        
        if self.enable_torch_compile:
            cmd.append("--enable-torch-compile")
        
        if self.disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")
            
        return cmd
    
    def check_health(self, timeout: int = 5, max_retries: int = None) -> bool:
        """Check SGLang server health - 参考原始健康检查逻辑"""
        if max_retries is None:
            max_retries = self.HEALTH_CHECK_MAX_RETRIES
            
        for retry in range(max_retries + 1):
            try:
                response = httpx.get(f"{self.server_url}/health", timeout=timeout)
                if response.status_code == 200:
                    if retry > 0:
                        logger.info(f"SGLang server health check succeeded after {retry + 1} attempts")
                    return True
                else:
                    logger.warning(f"SGLang server health check failed, status code: {response.status_code}")
            except Exception as e:
                if retry < max_retries:
                    delay = self.HEALTH_CHECK_RETRY_DELAYS[min(retry, len(self.HEALTH_CHECK_RETRY_DELAYS) - 1)]
                    logger.debug(f"SGLang server health check failed (attempt {retry + 1}/{max_retries + 1}): {e}")
                    logger.debug(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    logger.debug(f"SGLang server health check final failure: {e}")
        
        return False
    
    def wait_for_ready(self, timeout: int = 300, check_interval: int = 5) -> bool:
        """Wait for SGLang server to be ready - 参考原始等待逻辑"""
        logger.info(f"Waiting for SGLang server to be ready, max wait time: {timeout} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查进程是否还在运行
            if self.process and self.process.poll() is not None:
                logger.error("SGLang server process terminated unexpectedly")
                return False
            
            # 检查健康状态
            if self.check_health(timeout=5, max_retries=1):
                elapsed_time = time.time() - start_time
                logger.success(f"SGLang server started successfully! Took {elapsed_time:.1f} seconds")
                return True
                
            time.sleep(check_interval)
        
        logger.error(f"SGLang server failed to start within {timeout} seconds")
        return False
    
    def start_server(self) -> bool:
        """Start SGLang server - 完全参考原始启动逻辑"""
        with self._lock:
            # 检查是否已经有运行的服务器
            if self.check_health(timeout=3, max_retries=1):
                logger.info("SGLang server is already running")
                return True
            
            # 如果有旧进程，先停止
            if self.process:
                self.stop_server()
            
            try:
                logger.info(f"Starting SGLang server: {self.model_path} at {self.host}:{self.port}")
                
                # 构建启动命令
                cmd = self._build_command()
                logger.info(f"Executing command: {' '.join(cmd)}")
                
                # 设置环境变量 - 性能优化配置
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": "0",
                    "TRANSFORMERS_TRUST_REMOTE_CODE": "true" if self.trust_remote_code else "false",
                    "SGLANG_TRUST_REMOTE_CODE": "true" if self.trust_remote_code else "false",
                    "TOKENIZERS_PARALLELISM": "false",
                    # 🛡️ 保守内存管理环境变量
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.65",
                    "OMP_NUM_THREADS": "4",          # 减少 CPU 线程数，避免资源竞争
                    "MKL_NUM_THREADS": "4",          # 减少 Intel MKL 线程数
                })
                
                # 启动进程
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
                
                # 启动日志监控
                self._start_log_monitor()
                
                # 等待服务器就绪
                if self.wait_for_ready(timeout=300):
                    logger.info("SGLang server started successfully")
                    self._start_health_monitor()
                    return True
                else:
                    logger.error("SGLang server failed to start")
                    self.stop_server()
                    return False
                    
            except Exception as e:
                logger.exception(f"Failed to start SGLang server: {e}")
                self.stop_server()
                return False
    
    def stop_server(self) -> bool:
        """Stop SGLang server - 参考原始停止逻辑"""
        with self._lock:
            if not self.process:
                logger.info("SGLang server is not running")
                return True
            
            logger.info("Stopping SGLang server...")
            self.should_monitor = False
            
            try:
                # Graceful shutdown
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("SGLang server didn't shutdown gracefully within 5 seconds, forcing kill...")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                logger.info("SGLang server stopped successfully")
                return True
                
            except Exception as e:
                logger.exception(f"Error stopping SGLang server: {e}")
                return False
    
    def restart_server(self) -> bool:
        """Restart SGLang server"""
        logger.info("Restarting SGLang server...")
        if self.stop_server():
            time.sleep(2)  # 短暂等待
            return self.start_server()
        return False
    
    def is_running(self) -> bool:
        """Check if SGLang server process is running and healthy"""
        if not self.process:
            return False
        
        # Check if process is still alive
        if self.process.poll() is not None:
            return False
        
        # Check if server is responding to health checks
        return self.check_health(timeout=3, max_retries=1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get SGLang server status - 参考原始状态格式"""
        process_status = "running" if self.process and self.process.poll() is None else "stopped"
        health_status = self.check_health(timeout=3, max_retries=1) if process_status == "running" else False
        monitor_status = "running" if self.should_monitor and self.health_monitor_thread and self.health_monitor_thread.is_alive() else "stopped"
        
        return {
            "status": process_status,
            "health": health_status,
            "url": self.server_url,
            "model_path": self.model_path,
            "pid": self.process.pid if self.process else None,
            "monitor": {
                "status": monitor_status,
                "interval": 30  # 健康检查间隔
            }
        }
    
    def _start_log_monitor(self):
        """Start log monitoring thread - 参考原始日志监控"""
        if not self.process:
            return
        
        def monitor_logs():
            try:
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        # 输出所有日志，使用不同级别
                        if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                            logger.error(f"SGLang: {line}")
                        elif any(keyword in line.lower() for keyword in ['warning', 'warn', 'deprecated']):
                            logger.warning(f"SGLang: {line}")
                        elif any(keyword in line.lower() for keyword in ['ready', 'started', 'listening', 'server is ready']):
                            logger.success(f"SGLang: {line}")
                        elif any(keyword in line.lower() for keyword in ['loading', 'initializing', 'downloading']):
                            logger.info(f"SGLang: {line}")
                        else:
                            # 输出所有其他日志，但使用debug级别以免过于嘈杂
                            logger.debug(f"SGLang: {line}")
            except Exception as e:
                logger.debug(f"Log monitoring stopped: {e}")
        
        log_thread = threading.Thread(target=monitor_logs, daemon=True)
        log_thread.start()
    
    def _start_health_monitor(self):
        """Start health monitoring thread - 参考原始监控逻辑"""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            return
        
        self.should_monitor = True
        
        def health_monitor():
            consecutive_failures = 0
            max_failures = 3
            check_interval = int(os.getenv("SGLANG_MONITOR_INTERVAL", "10"))  # 参考原始配置
            
            while self.should_monitor:
                try:
                    if self.process and self.process.poll() is None:
                        # 进程还在运行，检查健康状态
                        if self.check_health(timeout=5, max_retries=1):
                            if consecutive_failures > 0:
                                logger.info("SGLang server health restored")
                                consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            logger.warning(f"SGLang server health check failed ({consecutive_failures}/{max_failures})")
                            
                            if consecutive_failures >= max_failures:
                                logger.error("SGLang server consecutive health check failures, attempting restart...")
                                if self.restart_server():
                                    consecutive_failures = 0
                                    logger.info("SGLang server restart successful")
                                else:
                                    logger.error("SGLang server restart failed")
                                    break
                    else:
                        # 进程已停止
                        if self.process:
                            logger.error("Detected SGLang server process stopped unexpectedly")
                            self.stop_server()
                        break
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    logger.exception(f"Health monitor error: {e}")
                    time.sleep(check_interval)
        
        self.health_monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        self.health_monitor_thread.start()
        logger.info(f"SGLang monitor thread started, check interval: {int(os.getenv('SGLANG_MONITOR_INTERVAL', '10'))} seconds")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up SGLang server resources...")
        self.should_monitor = False
        self.stop_server()


# Global SGLang server manager instance
_sglang_manager: Optional[SGLangServerManager] = None


def get_sglang_manager() -> SGLangServerManager:
    """Get global SGLang server manager instance"""
    global _sglang_manager
    if _sglang_manager is None:
        _sglang_manager = SGLangServerManager()
    return _sglang_manager


def start_sglang_server() -> bool:
    """Start SGLang server using global manager"""
    manager = get_sglang_manager()
    return manager.start_server()


def stop_sglang_server() -> bool:
    """Stop SGLang server using global manager"""
    manager = get_sglang_manager()
    return manager.stop_server()


def restart_sglang_server() -> bool:
    """Restart SGLang server using global manager"""
    manager = get_sglang_manager()
    return manager.restart_server()


def check_sglang_health() -> bool:
    """Check SGLang server health using global manager"""
    manager = get_sglang_manager()
    return manager.check_health()


def get_sglang_status() -> Dict[str, Any]:
    """Get SGLang server status using global manager"""
    manager = get_sglang_manager()
    return manager.get_status()


if __name__ == "__main__":
    # For testing purposes
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        manager = get_sglang_manager()
        
        if command == "start":
            success = manager.start_server()
            print(f"SGLang server start: {'Success' if success else 'Failed'}")
            sys.exit(0 if success else 1)
        elif command == "stop":
            success = manager.stop_server()
            print(f"SGLang server stop: {'Success' if success else 'Failed'}")
            sys.exit(0 if success else 1)
        elif command == "restart":
            success = manager.restart_server()
            print(f"SGLang server restart: {'Success' if success else 'Failed'}")
            sys.exit(0 if success else 1)
        elif command == "status":
            status = manager.get_status()
            print(f"SGLang Server Status: {status}")
            sys.exit(0 if status['status'] == 'running' else 1)
        elif command == "health":
            healthy = manager.check_health()
            print(f"SGLang Server Health: {'OK' if healthy else 'Failed'}")
            sys.exit(0 if healthy else 1)
    else:
        print("Usage: python sglang_manager.py [start|stop|restart|status|health]")
        sys.exit(1)