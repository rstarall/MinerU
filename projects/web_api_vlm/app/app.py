import json
import os
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from loguru import logger

from ocr import (
    init_writers, process_vlm_file_with_retry, MemoryDataWriter, 
    initialize_backend, get_current_backend, get_sglang_server_status,
    validate_backend, vlm_union_make_paginated,
)

from sglang_manager import start_sglang_server, get_sglang_status, check_sglang_health, stop_sglang_server
from process import (
    process_text_and_office_documents, process_image_files, process_pdf_files,
    pdf_extensions, office_extensions, image_extensions, text_extensions, all_supported_extensions,
    detect_file_type, detect_file_type_from_content, check_document_complexity
)
from mineru.utils.enum_class import MakeMode

app = FastAPI(
    title="MinerU VLM Web API",
    description="Multi-format file parsing API service based on MinerU 2.0 VLM model",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    # Start SGLang server first
    logger.info("Starting SGLang server...")
    sglang_started = start_sglang_server()
    
    if sglang_started:
        logger.info("SGLang server started successfully")
    else:
        logger.warning("SGLang server failed to start, will fallback to transformers backend")
    
    # Initialize backend (will detect SGLang server status and choose appropriate backend)
    initialize_backend()

@app.get("/health")
async def health_check():
    # Get SGLang status from both ocr module and sglang module
    sglang_server_status = get_sglang_server_status()  # From ocr module
    sglang_detailed_status = get_sglang_status()       # From sglang module
    
    return {
        "status": "healthy",
        "service": "mineru-vlm-api",
        "current_backend": get_current_backend(),
        "sglang_server": sglang_server_status,
        "sglang_detailed": sglang_detailed_status
    }

@app.post(
    "/vlm_parse",
    tags=["VLM Processing"],
    summary="Parse documents using VLM model",
    description="Parse PDF, Office documents or image files using Vision Language Model"
)
async def vlm_parse(
    file: UploadFile,
    backend: str = Form(None, description="VLM backend type: sglang-client, transformers"),
    return_images: bool = Form(False, description="Return image information"),
    return_middle_json: bool = Form(False, description="Return intermediate JSON"),
    return_paginated_md: bool = Form(False, description="Return paginated markdown content")
):
    """Parse multi-format documents using VLM model"""
    try:
        file_name = os.path.basename(file.filename).split(".")[0] if file.filename else "uploaded_file"
        
        # Read file and detect type
        file_bytes = await file.read()
        file_extension = detect_file_type_from_content(file_bytes)
        
        if not file_extension:
            return JSONResponse(
                content={"error": "Cannot determine file type"},
                status_code=400
            )

        # Use memory writer for images (no file saving)
        image_writer = MemoryDataWriter()
        
        validated_backend = validate_backend(backend) if backend else get_current_backend()
        
        # Calculate complexity info for VLM processing
        complexity_info = check_document_complexity(file_bytes, file_extension)

        middle_json, infer_result, md_content = await process_vlm_file_with_retry(
            file_bytes=file_bytes,
            file_extension=file_extension,
            image_writer=image_writer,
            backend=validated_backend,
            max_retries=3,
            complexity_info=complexity_info
        )

        file_type = detect_file_type(file_extension)

        data = {
            "md_content": md_content,
            "file_name": file_name,
            "file_type": file_type,
            "file_extension": file_extension,
            "backend": validated_backend
        }

        if return_paginated_md:
            pdf_info = middle_json["pdf_info"]
            paginated_md = vlm_union_make_paginated(pdf_info, MakeMode.MM_MD, "images")
            data["md_content_pages"] = paginated_md
            data["total_pages"] = len(paginated_md)

        if return_middle_json:
            data["middle_json"] = middle_json

        if return_images:
            binary_files = getattr(image_writer, 'binary_files', {})
            data["images"] = {
                "count": len(binary_files),
                "files": list(binary_files.keys()),
                "message": "Images processed in memory"
            }

        return JSONResponse(content=data, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VLM processing failed: {str(e)}")
        return JSONResponse(
            content={"error": f"File processing error: {str(e)}"},
            status_code=500
        )


@app.post(
    "/process",
    tags=["Text Extraction"],
    summary="Extract text using OCR",
    description="Extract text content from PDF, Office documents or images"
)
@app.put(
    "/process",
    tags=["Text Extraction"],
    summary="Extract text using OCR",
    description="Extract text content from PDF, Office documents or images"
)
async def process_document(
    request: Request,
    file: UploadFile = None,
    lang: str = Form("ch", description="Document language: ch, en, ja, ko"),
    timeout: int = Form(None, description="Processing timeout in seconds")
):
    """Text extraction API using VLM"""
    try:
        # Get file data
        if request.method == "PUT":
            file_bytes = await request.body()
            if not file_bytes:
                return JSONResponse(
                    content={"error": "No file data provided in request body"},
                    status_code=400
                )
            
            # Extract filename from headers (more efficient)
            x_filename = request.headers.get("X-Filename")
            content_disposition = request.headers.get("Content-Disposition")
            logger.info(f"PUT request headers - X-Filename: {x_filename}, Content-Disposition: {content_disposition}")
            
            filename = x_filename or content_disposition or ""
            if "filename=" in filename:
                filename = filename.split("filename=")[-1].strip('"')
            if not filename:
                filename = "uploaded_file"
                
            logger.info(f"Final extracted filename: {filename}")
                
        else:
            if file is None:
                return JSONResponse(
                    content={"error": "File is required"},
                    status_code=400
                )
            file_bytes = await file.read()
            filename = file.filename or "uploaded_file"
        
        # Try to get extension from filename first (most efficient)
        file_extension = os.path.splitext(filename)[1].lower() if filename else ""
        logger.info(f"Filename: {filename}, Extracted extension: {file_extension}")
        
        if file_extension and file_extension in all_supported_extensions:
            # Have valid extension, use it directly (highest efficiency)
            logger.info(f"Using extension from filename: {file_extension}")
            pass
        else:
            # No extension or unsupported extension, use content detection
            logger.info(f"No valid extension, detecting from content. File size: {len(file_bytes)} bytes")
            logger.info(f"First 100 bytes: {file_bytes[:100]}")
            file_extension = detect_file_type_from_content(file_bytes)
            logger.info(f"Detected extension from content: {file_extension}")
            if not file_extension or file_extension not in all_supported_extensions:
                return JSONResponse(
                    content={"error": f"Unsupported file format. Supported formats: {all_supported_extensions}"},
                    status_code=400
                )
        
        # Generate consistent filename
        file_name = os.path.splitext(os.path.basename(filename))[0] or "uploaded_file"
        filename = f"{file_name}{file_extension}"
        
        logger.info(f"Final processing: filename={filename}, extension={file_extension}, file_size={len(file_bytes)}")
        
        supported_langs = ["ch", "en", "ja", "ko"]
        if lang not in supported_langs:
            lang = "ch"

        # 根据文件类型自动选择处理函数
        if file_extension in text_extensions or file_extension in office_extensions:
            # 处理文本文件和Office文档
            extracted_text = await process_text_and_office_documents(
                file_bytes=file_bytes,
                file_extension=file_extension,
                timeout=timeout
            )
        elif file_extension in image_extensions:
            # 处理图像文件
            extracted_text = await process_image_files(
                file_bytes=file_bytes,
                file_extension=file_extension,
                timeout=timeout
            )
        elif file_extension in pdf_extensions:
            # 处理PDF文件
            extracted_text = await process_pdf_files(
                file_bytes=file_bytes,
                file_extension=file_extension,
                timeout=timeout
            )
        else:
            # 不支持的文件类型
            logger.error(f"Unsupported file type: {file_extension}")
            return JSONResponse(
                content={"error": f"Unsupported file type: {file_extension}. Supported formats: {all_supported_extensions}"},
                status_code=400
            )
        
        try:
            # Estimate page count based on file size (more efficient than parsing PDF)
            file_size_mb = len(file_bytes) / (1024 * 1024)
            if file_extension in pdf_extensions:
                # Rough estimation: 1 page ≈ 0.5-2MB depending on content
                page_count = max(1, int(file_size_mb / 1.5))
            elif file_extension in image_extensions:
                page_count = 1
            elif file_extension in office_extensions:
                # Rough estimation for office docs
                page_count = max(1, int(file_size_mb / 0.5))
            elif file_extension in text_extensions:
                page_count = 1
            else:
                page_count = 1
        except Exception as e:
            logger.warning(f"Cannot estimate page count: {e}")
            page_count = 1
        
        lang_mapping = {
            "ch": "zh-CN",
            "en": "en-US", 
            "ja": "ja-JP",
            "ko": "ko-KR"
        }
        
        response_data = {
            "page_content": extracted_text,
            "metadata": {
                "source": f"{file_name}{file_extension}",
                "page_count": page_count,
                "language": lang_mapping.get(lang, "zh-CN"),
                "extraction_engine": "MinerU-VLM",
                "file_type": detect_file_type(file_extension),
                "file_size_bytes": len(file_bytes),
                "processing_backend": get_current_backend()
            }
        }
        
        return JSONResponse(content=response_data, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VLM text extraction failed: {str(e)}")
        return JSONResponse(
            content={"error": f"VLM text extraction failed: {str(e)}"},
            status_code=500
        )

# 简化的SGLang状态查询API（仅用于监控和调试）
@app.get(
    "/sglang/status",
    tags=["System Status"],
    summary="Get SGLang server status",
    description="Get current SGLang server status and backend information"
)
async def get_sglang_status_api():
    """Get SGLang server status for monitoring and debugging"""
    try:
        return JSONResponse(
            content={
                "sglang_status": get_sglang_status(),
                "current_backend": get_current_backend(),
                "health_check": check_sglang_health()
            },
            status_code=200
        )
    except Exception as e:
        logger.exception(f"Failed to get SGLang status: {str(e)}")
        return JSONResponse(
            content={
                "error": f"Failed to get SGLang status: {str(e)}"
            },
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )