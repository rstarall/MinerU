import json
import os
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

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.data.data_reader_writer import DataWriter, FileBasedDataWriter
from mineru.data.data_reader_writer.s3 import S3DataWriter
from mineru.utils.config_reader import get_bucket_name, get_s3_config
from mineru.utils.enum_class import MakeMode



from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze, aio_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

import mineru.model.vlm_hf_model
import mineru.model.vlm_sglang_model

# check_document_complexity is now handled via parameter passing to avoid circular import

# File extensions have been migrated to process.py

# Global variables for VLM backend
CURRENT_BACKEND = "sglang-client"
SGLANG_SERVER_URL = "http://localhost:30000"
SGLANG_MODEL_PATH = None

# Timeout settings
DEFAULT_VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))

# Server state lock
SERVER_STATE_LOCK = threading.RLock()

def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message with more useful descriptions"""
    error_msg = str(error) if str(error) else "Unknown error"
    error_type = str(type(error))

    if "PdfiumError" in error_type or "data format error" in error_msg.lower():
        return "Invalid file format. The uploaded file is not a valid document or is corrupted."
    elif "connection" in error_msg.lower() or "server" in error_msg.lower():
        return f"Server connection failed: {error_msg}"
    elif "model" in error_msg.lower() and "load" in error_msg.lower():
        return f"Model loading failed: {error_msg}"
    elif "memory" in error_msg.lower() or "cuda" in error_msg.lower():
        return f"Memory or GPU resource insufficient: {error_msg}"
    elif "format" in error_msg.lower() or "decode" in error_msg.lower():
        return f"File format error: {error_msg}"
    elif "timeout" in error_msg.lower():
        return f"Processing timeout: {error_msg}"
    elif not error_msg or error_msg == "Unknown error":
        return f"{context} failed: Unknown error occurred, check logs for details"
    else:
        return f"{context}: {error_msg}"

def check_sglang_server_health(url: str) -> bool:
    """Check SGLang server health"""
    try:
        response = httpx.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def cleanup_resources():
    """Cleanup resources"""
    logger.info("Cleaning up resources...")

def signal_handler(signum, frame):
    cleanup_resources()
    exit(0)

class MemoryDataWriter(DataWriter):
    """Memory data writer for temporarily storing data in memory"""

    def __init__(self):
        self.buffer = StringIO()
        self.binary_files = {}

    def write(self, filename: str, content):
        """Implement abstract method write"""
        if isinstance(content, str):
            self.write_string(filename, content)
        elif isinstance(content, bytes):
            self.write_binary(filename, content)
        elif isinstance(content, list):
            self.write_jsonl(filename, content)
        else:
            self.write_string(filename, str(content))

    def write_string(self, filename: str, content: str):
        """Write string content"""
        self.buffer.write(f"=== {filename} ===\n")
        self.buffer.write(content)
        self.buffer.write("\n\n")

    def write_jsonl(self, filename: str, content: list):
        """Write JSONL format content"""
        self.buffer.write(f"=== {filename} ===\n")
        for item in content:
            self.buffer.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.buffer.write("\n")

    def write_binary(self, filename: str, content: bytes):
        """Write binary content"""
        self.binary_files[filename] = content

    def get_value(self) -> str:
        """Get all text content"""
        return self.buffer.getvalue()

    def get_binary(self, filename: str) -> bytes:
        """Get specified binary file content"""
        return self.binary_files.get(filename, b"")

    def close(self):
        """Close buffer"""
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
    """Initialize data writers and get file content"""
    if file is not None:
        file_bytes = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    else:
        raise ValueError("File is required")
    
    # Initialize writers
    if output_path and output_path.startswith("s3://"):
        bucket = get_bucket_name(output_path)
        ak, sk, endpoint = get_s3_config(output_path)
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
        writer = FileBasedDataWriter(output_path)
        if output_image_path:
            image_writer = FileBasedDataWriter(output_image_path)
        else:
            image_writer = MemoryDataWriter()
    else:
        writer = MemoryDataWriter()
        image_writer = MemoryDataWriter()

    return writer, image_writer, file_bytes, file_extension

# File type detection functions have been migrated to process.py





def initialize_backend():
    """Initialize VLM backend"""
    global CURRENT_BACKEND, SGLANG_MODEL_PATH
    
    with SERVER_STATE_LOCK:
        try:
            if check_sglang_server_health(SGLANG_SERVER_URL):
                CURRENT_BACKEND = "sglang-client"
                logger.info("Using sglang-client backend")
            else:
                CURRENT_BACKEND = "transformers"
                SGLANG_MODEL_PATH = auto_download_and_get_model_root_path("/", "vlm")
                logger.info("Using transformers backend")
        except Exception as e:
            logger.warning(f"Backend initialization failed: {e}, using transformers")
            CURRENT_BACKEND = "transformers"
            SGLANG_MODEL_PATH = auto_download_and_get_model_root_path("/", "vlm")

def get_current_backend():
    """Get current backend"""
    with SERVER_STATE_LOCK:
        return CURRENT_BACKEND

def get_sglang_server_status():
    """Get SGLang server status"""
    with SERVER_STATE_LOCK:
        sglang_health = check_sglang_server_health(SGLANG_SERVER_URL)
        return {
            "status": "running" if sglang_health else "stopped",
            "health": sglang_health,
            "url": SGLANG_SERVER_URL
        }

def validate_backend(backend: str) -> str:
    """Validate and normalize VLM backend type"""
    if backend is None:
        return get_current_backend()
    
    valid_backends = ["sglang-client", "transformers"]
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend: {backend}. Supported backends: {valid_backends}")
    
    return backend

# check_document_complexity function has been migrated to process.py

async def process_vlm_file(
    file_bytes: bytes,
    file_extension: str,
    image_writer,
    backend: str = None,
    timeout: int = None,
    complexity_info: dict = None
) -> Tuple[dict, list, str]:
    """Process file using VLM"""
    # File validation is done in app.py, proceed with processing

    effective_backend = validate_backend(backend)
    effective_timeout = timeout or (complexity_info or {}).get("recommended_timeout", DEFAULT_VLM_TIMEOUT)

    try:
        vlm_kwargs = {}
        if effective_backend == "sglang-client":
            vlm_kwargs["server_url"] = SGLANG_SERVER_URL
        elif effective_backend == "transformers":
            vlm_kwargs["model_path"] = SGLANG_MODEL_PATH or auto_download_and_get_model_root_path("/", "vlm")

        start_time = time.time()

        try:
            middle_json = await asyncio.wait_for(
                aio_doc_analyze(
                    file_bytes,
                    image_writer=image_writer,
                    backend=effective_backend,
                    **vlm_kwargs
                ),
                timeout=effective_timeout
            )

            elapsed_time = time.time() - start_time
            logger.info(f"VLM analysis completed in {elapsed_time:.1f}s")

        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.error(f"VLM analysis timeout ({effective_timeout}s), actual time: {elapsed_time:.1f}s")
            raise HTTPException(
                status_code=408,
                detail=f"VLM analysis timeout ({effective_timeout}s), try reducing document pages or retry later"
            )

        except Exception as vlm_error:
            logger.exception(f"VLM analysis failed: {vlm_error}")
            
            # Handle specific error types with better messages
            error_str = str(vlm_error)
            if "PdfiumError" in str(type(vlm_error)) or "Data format error" in error_str:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. The file appears to be corrupted or not a valid PDF/document format."
                )
            elif "Failed to load document" in error_str:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot load document. Please check if the file is a valid document format and not corrupted."
                )
            
            detail = format_error_message(vlm_error, "VLM analysis")
            raise HTTPException(status_code=500, detail=detail)

        # Generate markdown content
        try:
            # Handle case where middle_json might be a tuple instead of dict
            if isinstance(middle_json, tuple):
                logger.warning(f"middle_json is a tuple with {len(middle_json)} elements, extracting first element")
                actual_json = middle_json[0]
            else:
                actual_json = middle_json
            
            pdf_info = actual_json["pdf_info"]
            infer_result = []
            
            for page_info in pdf_info:
                for block in page_info.get("blocks", []):
                    infer_result.append(block)
            
            # Generate complete markdown
            md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, "")
            
            return actual_json, infer_result, md_content

        except Exception as content_error:
            logger.exception(f"Content generation failed: {content_error}")
            detail = format_error_message(content_error, "Content generation")
            raise HTTPException(status_code=500, detail=detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VLM processing failed: {e}")
        detail = format_error_message(e, "VLM processing")
        raise HTTPException(status_code=500, detail=detail)

async def process_vlm_file_with_retry(
    file_bytes: bytes,
    file_extension: str,
    image_writer,
    backend: str = None,
    timeout: int = None,
    max_retries: int = 3,
    complexity_info: dict = None
) -> Tuple[dict, list, str]:
    """Process VLM file with retry mechanism"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await process_vlm_file(
                file_bytes=file_bytes,
                file_extension=file_extension,
                image_writer=image_writer,
                backend=backend,
                timeout=timeout,
                complexity_info=complexity_info
            )
        except HTTPException as e:
            if e.status_code == 408:  # Timeout error
                raise  # Don't retry timeout errors
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"VLM processing attempt {attempt + 1} failed, retrying in {wait_time}s: {e.detail}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"VLM processing failed after {max_retries} attempts")
                raise
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"VLM processing attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"VLM processing failed after {max_retries} attempts")
                detail = format_error_message(e, "VLM processing")
                raise HTTPException(status_code=500, detail=detail)
    
    # This should never be reached, but just in case
    if last_exception:
        if isinstance(last_exception, HTTPException):
            raise last_exception
        else:
            detail = format_error_message(last_exception, "VLM processing")
            raise HTTPException(status_code=500, detail=detail)

def vlm_union_make_paginated(pdf_info_dict: list, make_mode: str, img_buket_path: str = '') -> list:
    """Generate paginated markdown content"""
    page_md_list = []
    
    for page_info in pdf_info_dict:
        if 'blocks' in page_info:
            page_content = vlm_union_make([page_info], make_mode, img_buket_path)
            page_md_list.append(page_content)

    return page_md_list

# Register cleanup function for program exit
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)