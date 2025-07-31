import json
import os
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from loguru import logger

# 从 ocr.py 导入核心功能函数
from ocr import (
    # 文件扩展名常量
    pdf_extensions,
    office_extensions,
    image_extensions,
    text_extensions,
    all_supported_extensions,

    # 核心功能函数
    init_writers,
    detect_file_type,
    process_vlm_file_with_retry,
    ocr_extract,
    MemoryDataWriter,
    check_document_complexity,

    # VLM和SGLang相关功能
    initialize_backend,
    get_current_backend,
    get_sglang_server_status,
    validate_backend,
    vlm_union_make_paginated,
    start_monitor,
    stop_monitor,
    restart_sglang_server,
    set_monitor_interval,
    get_monitor_interval,
    set_health_check_retries,
    get_health_check_config,
)

# 从 MinerU 导入相关模块
from mineru.utils.enum_class import MakeMode
from mineru.utils.pdf_image_tools import load_images_from_pdf
from mineru.data.data_reader_writer.s3 import S3DataReader
from mineru.utils.config_reader import get_bucket_name, get_s3_config




app = FastAPI(
    title="MinerU VLM Web API",
    description="基于MinerU 2.0 VLM模型的多格式文件解析API服务",
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """应用启动事件：初始化VLM后端"""
    initialize_backend()







@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "MinerU VLM Web API",
        "version": "2.0.0",
        "description": "基于MinerU 2.0 VLM模型的多格式文件解析API服务",
        "current_backend": get_current_backend(),
        "supported_backends": ["sglang-client", "transformers"],
        "supported_formats": {
            "pdf": pdf_extensions,
            "office": office_extensions,
            "image": image_extensions
        },
        "features": {
            "vlm_parsing": "支持VLM模型进行公式和表格识别",
            "pagination": "支持按原始页面分页输出Markdown内容",
            "storage": "支持内存处理和S3存储",
            "formats": "支持PDF、Office文档和图像文件",
            "text_extraction": "支持纯文本OCR提取功能"
        },
        "endpoints": {
            "/vlm_parse": "VLM模型文档解析，支持Markdown输出和结构化数据",
            "/process": "VLM模型高级文档解析，返回纯Markdown文本内容",
            "/health": "健康检查接口",
            "/docs": "API文档"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    sglang_server_status = get_sglang_server_status()

    return {
        "status": "healthy",
        "service": "mineru-vlm-api",
        "current_backend": get_current_backend(),
        "sglang_server": sglang_server_status
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
        validated_backend = validate_backend(backend) if backend else get_current_backend()

        # 处理文件（使用带重试机制的函数）
        middle_json, infer_result, md_content = await process_vlm_file_with_retry(
            file_bytes=file_bytes,
            file_extension=file_extension,
            image_writer=image_writer,
            backend=validated_backend,
            max_retries=3  # 默认重试3次
        )

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
                binary_files = getattr(image_writer, 'binary_files', {})
                data["images"] = {
                    "count": len(binary_files),
                    "files": list(binary_files.keys()),
                    "message": "图像已在内存中处理"
                }
            else:
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


@app.post(
    "/process",
    tags=["OCR文本提取"],
    summary="OCR文本提取",
    description="从任意文档文件（PDF、Office文档、图像）中提取纯文本内容"
)

@app.put(
    "/process",
    tags=["OCR文本提取"],
    summary="OCR文本提取",
    description="从任意文档文件（PDF、Office文档、图像）中提取纯文本内容"
)
async def process_document(
    request: Request,
    file: UploadFile = None,
    file_path: str = Form(None, description="文件路径（本地路径或S3路径）"),
    lang: str = Form("ch", description="文档语言，支持: ch(中文), en(英文)等"),
    timeout: int = Form(None, description="处理超时时间（秒），默认根据文档复杂度自动设置")
):
    """
    OCR文本提取API

    Args:
        request: 请求对象
        file: 上传的文件（PDF、Office文档或图像）
        file_path: 文件路径（本地路径或S3路径）
        lang: 文档语言设置

    Returns:
        包含提取文本和元数据的JSON响应
    """
    try:
        # 检查是否是PUT请求
        if request.method == "PUT":
            if file:
                # 如果提供了file参数，使用file.read()
                file_bytes = await file.read()
                filename = file.filename or "uploaded_file"
                file_extension = os.path.splitext(filename)[1].lower() if filename else ""
                file_name = os.path.basename(filename).split(".")[0] if filename else "uploaded_file"
            else:
                # 如果没有提供file参数，尝试从请求体读取
                try:
                    file_bytes = await request.body()
                    # 从请求头获取文件名
                    filename = request.headers.get("X-Filename", "uploaded_file")
                    file_extension = os.path.splitext(filename)[1].lower() if filename else ""
                    file_name = os.path.basename(filename).split(".")[0] if filename else "uploaded_file"
                except RuntimeError as e:
                    # 如果请求体已被消费，返回错误
                    return JSONResponse(
                        content={"error": f"无法读取请求体: {str(e)}"},
                        status_code=400
                    )
        else:
            # POST请求处理逻辑
            if file is None and file_path is None:
                return JSONResponse(
                    content={"error": "必须提供file或file_path中的一个"},
                    status_code=400
                )
            
            if file:
                file_bytes = await file.read()
                file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
                file_name = os.path.basename(file.filename).split(".")[0] if file.filename else "uploaded_file"
            else:
                # 处理file_path
                # 保持原有逻辑不变
                file_extension = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path).split(".")[0]

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
        
        if not file_extension:
            return JSONResponse(
                content={"error": "无法确定文件类型"},
                status_code=400
            )
        
        # 验证语言参数（VLM不需要语言参数，但保持接口兼容性）
        supported_langs = ["ch", "en", "ja", "ko"]  # 支持的语言
        if lang not in supported_langs:
            lang = "ch"  # 默认使用中文

        # 初始化图像写入器（VLM需要）
        image_writer = MemoryDataWriter()

        # 使用VLM进行高级文档解析 - 获取完整的markdown内容（使用带重试机制的函数）
        _, _, md_content = await process_vlm_file_with_retry(
            file_bytes=file_bytes,
            file_extension=file_extension,
            image_writer=image_writer,
            backend=None,  # 使用默认后端
            timeout=timeout,  # 传递超时参数
            max_retries=3  # 默认重试3次
        )

        # 使用VLM生成的markdown内容作为提取的文本
        extracted_text = md_content if md_content else ""
        
        # 计算页面数量
        try:
            if file_extension in pdf_extensions:
                # 对于PDF文件，可以通过加载PDF文档来获取页数
                _, pdf_doc = load_images_from_pdf(file_bytes)
                page_count = len(pdf_doc)  # 获取PDF文档的页面数量
                # 确保释放pdf_doc资源
                try:
                    pdf_doc.close()
                except Exception:
                    pass
            elif file_extension in image_extensions:
                # 图像文件只有1页
                page_count = 1
            elif file_extension in office_extensions:
                # Office文件暂时无法直接获取页数，设置为1
                page_count = 1
            elif file_extension in text_extensions:
                # 文本文件只有1页
                page_count = 1
            else:
                page_count = 1
        except Exception as e:
            logger.warning(f"无法获取页面数量: {e}")
            page_count = 1
        
        # 语言代码映射
        lang_mapping = {
            "ch": "zh-CN",
            "en": "en-US", 
            "ja": "ja-JP",
            "ko": "ko-KR"
        }
        
        # 构建返回数据
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
        
        logger.info(f"成功使用VLM提取文件文本: {file_name} ({file_extension})，语言: {lang}，页数: {page_count}")
        return JSONResponse(
            content=response_data,
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"VLM文本提取过程中发生意外错误: {e}")
        return JSONResponse(
            content={"error": f"VLM文本提取失败: {str(e)}"},
            status_code=500
        )


@app.get("/monitor/status")
async def get_monitor_status():
    """获取SGLang服务器和监控状态"""
    try:
        status = get_sglang_server_status()
        return JSONResponse(content=status, status_code=200)
    except Exception as e:
        logger.exception("获取监控状态失败")
        return JSONResponse(
            content={"error": f"获取监控状态失败: {str(e)}"},
            status_code=500
        )


@app.post("/monitor/restart")
async def restart_server():
    """手动重启SGLang服务器"""
    try:
        logger.info("收到手动重启SGLang服务器请求")
        restart_sglang_server()
        return JSONResponse(
            content={"message": "SGLang服务器重启请求已发送"},
            status_code=200
        )
    except Exception as e:
        logger.exception("重启SGLang服务器失败")
        return JSONResponse(
            content={"error": f"重启SGLang服务器失败: {str(e)}"},
            status_code=500
        )


@app.get("/monitor/health-check-config")
async def get_health_check_configuration():
    """获取健康检查配置"""
    try:
        config = get_health_check_config()
        return JSONResponse(content=config, status_code=200)
    except Exception as e:
        logger.exception("获取健康检查配置失败")
        return JSONResponse(
            content={"error": f"获取健康检查配置失败: {str(e)}"},
            status_code=500
        )


@app.post("/monitor/health-check-config")
async def update_health_check_configuration(
    max_retries: int = Form(..., description="最大重试次数 (1-10)"),
    retry_delays: str = Form(None, description="重试延迟时间列表，用逗号分隔，如: 1,2,3")
):
    """更新健康检查配置"""
    try:
        # 解析重试延迟时间
        delays = None
        if retry_delays:
            try:
                delays = [float(d.strip()) for d in retry_delays.split(',') if d.strip()]
            except ValueError:
                return JSONResponse(
                    content={"error": "重试延迟时间格式错误，请使用逗号分隔的数字，如: 1,2,3"},
                    status_code=400
                )

        # 更新配置
        set_health_check_retries(max_retries, delays)

        # 返回更新后的配置
        new_config = get_health_check_config()
        logger.info(f"健康检查配置已更新: {new_config}")

        return JSONResponse(
            content={
                "message": "健康检查配置更新成功",
                "config": new_config
            },
            status_code=200
        )
    except ValueError as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )
    except Exception as e:
        logger.exception("更新健康检查配置失败")
        return JSONResponse(
            content={"error": f"更新健康检查配置失败: {str(e)}"},
            status_code=500
        )


@app.post("/monitor/start")
async def start_monitoring():
    """启动SGLang监控"""
    try:
        start_monitor()
        return JSONResponse(
            content={"message": "SGLang监控已启动"},
            status_code=200
        )
    except Exception as e:
        logger.exception("启动监控失败")
        return JSONResponse(
            content={"error": f"启动监控失败: {str(e)}"},
            status_code=500
        )


@app.post("/monitor/stop")
async def stop_monitoring():
    """停止SGLang监控"""
    try:
        stop_monitor()
        return JSONResponse(
            content={"message": "SGLang监控已停止"},
            status_code=200
        )
    except Exception as e:
        logger.exception("停止监控失败")
        return JSONResponse(
            content={"error": f"停止监控失败: {str(e)}"},
            status_code=500
        )


@app.get("/monitor/interval")
async def get_monitoring_interval():
    """获取当前监控间隔"""
    try:
        interval = get_monitor_interval()
        return JSONResponse(
            content={"interval": interval, "unit": "seconds"},
            status_code=200
        )
    except Exception as e:
        logger.exception("获取监控间隔失败")
        return JSONResponse(
            content={"error": f"获取监控间隔失败: {str(e)}"},
            status_code=500
        )


@app.post("/monitor/interval")
async def set_monitoring_interval(interval: int = Form(...)):
    """设置监控间隔"""
    try:
        set_monitor_interval(interval)
        return JSONResponse(
            content={"message": f"监控间隔已设置为{interval}秒"},
            status_code=200
        )
    except ValueError as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=400
        )
    except Exception as e:
        logger.exception("设置监控间隔失败")
        return JSONResponse(
            content={"error": f"设置监控间隔失败: {str(e)}"},
            status_code=500
        )


# 删除复杂的process-simple接口，简化代码


# 删除复杂的头部清理中间件，简化代码


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    ) 
