import json
import os
import tempfile
from base64 import b64encode
from glob import glob
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

# Import MinerU 2.0 modules
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode

app = FastAPI(
    title="MinerU VLM Web API",
    description="基于MinerU 2.0 VLM模型的PDF解析API服务",
    version="2.0.0"
)

# Supported file extensions
pdf_extensions = [".pdf"]
image_extensions = [".png", ".jpg", ".jpeg"]


class MemoryDataWriter:
    """内存数据写入器，用于在内存中暂存数据"""
    
    def __init__(self):
        self.buffer = StringIO()
        self.data = {}

    def write(self, path: str, data: bytes) -> None:
        """写入二进制数据"""
        if isinstance(data, str):
            self.data[path] = data
        else:
            self.data[path] = data.decode("utf-8")

    def write_string(self, path: str, data: str) -> None:
        """写入字符串数据"""
        self.data[path] = data

    def get_value(self, path: str = None) -> str:
        """获取指定路径的数据，如果未指定路径则返回buffer内容"""
        if path:
            return self.data.get(path, "")
        return self.buffer.getvalue()

    def close(self):
        """关闭buffer"""
        self.buffer.close()


def validate_vlm_backend(backend: str) -> str:
    """验证并规范化VLM后端名称"""
    valid_backends = ["vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"]
    
    # 如果用户传入的是简化版本，转换为完整名称
    if backend in ["transformers", "sglang-engine", "sglang-client"]:
        backend = f"vlm-{backend}"
    
    if backend not in valid_backends:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported backend: {backend}. Supported backends: {valid_backends}"
        )
    
    return backend


def encode_image(image_path: str) -> str:
    """将图像文件编码为base64字符串"""
    try:
        with open(image_path, "rb") as f:
            return b64encode(f.read()).decode()
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return ""


def process_vlm_file(
    file_bytes: bytes,
    file_extension: str,
    backend: str,
    server_url: Optional[str] = None,
    start_page_id: int = 0,
    end_page_id: Optional[int] = None,
    output_dir: str = "output",
    file_name: str = "document"
) -> Tuple[dict, List[str], str, str]:
    """
    使用VLM后端处理文件
    
    Args:
        file_bytes: 文件的二进制内容
        file_extension: 文件扩展名
        backend: VLM后端类型
        server_url: sglang服务器URL（仅当backend为vlm-sglang-client时需要）
        start_page_id: 开始页码（从0开始）
        end_page_id: 结束页码（从0开始，None表示处理到最后一页）
        output_dir: 输出目录
        file_name: 文件名（不含扩展名）
    
    Returns:
        Tuple[middle_json, infer_result, md_content, image_dir]: 中间JSON、推理结果、MD内容、图像目录
    """
    # 只支持PDF和图像文件
    if file_extension not in pdf_extensions + image_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. VLM backend only supports PDF and image files."
        )
    
    # 验证VLM后端
    backend = validate_vlm_backend(backend)
    
    # 准备输出目录
    local_image_dir, local_md_dir = prepare_env(output_dir, file_name, "vlm")
    image_writer = FileBasedDataWriter(local_image_dir)
    
    # 处理页面范围
    if file_extension in pdf_extensions:
        # 对于PDF文件，处理页面范围
        file_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
            file_bytes, start_page_id, end_page_id
        )
    
    # 移除vlm-前缀以匹配mineru的后端参数
    backend_param = backend[4:] if backend.startswith("vlm-") else backend
    
    try:
        # 使用VLM分析文档
        middle_json, infer_result = vlm_doc_analyze(
            file_bytes,
            image_writer=image_writer,
            backend=backend_param,
            server_url=server_url
        )
        
        # 生成MD内容
        pdf_info = middle_json["pdf_info"]
        image_dir = os.path.basename(local_image_dir)
        md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        
        return middle_json, infer_result, md_content, local_image_dir
        
    except Exception as e:
        logger.error(f"VLM processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"VLM processing failed: {str(e)}")


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "MinerU VLM Web API",
        "version": "2.0.0",
        "description": "基于MinerU 2.0 VLM模型的PDF解析API服务",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "mineru-vlm-api"}


@app.post(
    "/vlm_parse",
    tags=["VLM解析"],
    summary="使用VLM模型解析PDF或图像文件",
    description="使用视觉语言模型(VLM)解析PDF或图像文件，支持transformers、sglang-engine、sglang-client后端"
)
async def vlm_parse(
    file: UploadFile = None,
    file_path: str = Form(None, description="文件路径（与file参数二选一）"),
    backend: str = Form("vlm-transformers", description="VLM后端类型: vlm-transformers, vlm-sglang-engine, vlm-sglang-client"),
    server_url: Optional[str] = Form(None, description="SGLang服务器URL（仅当backend为vlm-sglang-client时需要）"),
    start_page: int = Form(0, description="开始页码（从0开始）", ge=0),
    end_page: Optional[int] = Form(None, description="结束页码（从0开始，None表示处理到最后一页）", ge=0),
    output_dir: str = Form("output", description="输出目录"),
    return_images: bool = Form(False, description="是否返回图像的base64编码"),
    return_middle_json: bool = Form(False, description="是否返回中间JSON格式"),
    return_model_output: bool = Form(False, description="是否返回模型原始输出"),
    save_files: bool = Form(False, description="是否保存文件到磁盘")
):
    """
    使用VLM模型解析PDF或图像文件
    
    支持的后端类型：
    - vlm-transformers: 使用transformers库的VLM模型（通用性好）
    - vlm-sglang-engine: 使用SGLang引擎的VLM模型（速度快）
    - vlm-sglang-client: 连接到SGLang服务器的客户端（需要单独启动服务器）
    
    返回数据格式：
    - md_content: Markdown格式的解析结果（始终返回）
    - images: 图像的base64编码（可选）
    - middle_json: 中间JSON格式数据（可选）
    - model_output: 模型原始输出（可选）
    """
    try:
        # 验证输入参数
        if (file is None and file_path is None) or (file is not None and file_path is not None):
            raise HTTPException(
                status_code=400,
                detail="必须提供file或file_path参数中的一个，但不能同时提供两个"
            )
        
        if backend == "vlm-sglang-client" and not server_url:
            raise HTTPException(
                status_code=400,
                detail="使用vlm-sglang-client后端时必须提供server_url参数"
            )
        
        if end_page is not None and end_page <= start_page:
            raise HTTPException(
                status_code=400,
                detail="end_page必须大于start_page"
            )
        
        # 获取文件内容和文件名
        if file:
            file_bytes = await file.read()
            file_name = Path(file.filename).stem if file.filename else "uploaded_file"
            file_extension = Path(file.filename).suffix.lower() if file.filename else ""
        else:
            # 处理文件路径
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            file_name = Path(file_path).stem
            file_extension = Path(file_path).suffix.lower()
        
        if not file_extension:
            raise HTTPException(status_code=400, detail="无法确定文件类型")
        
        # 处理文件
        middle_json, infer_result, md_content, image_dir = process_vlm_file(
            file_bytes=file_bytes,
            file_extension=file_extension,
            backend=backend,
            server_url=server_url,
            start_page_id=start_page,
            end_page_id=end_page,
            output_dir=output_dir,
            file_name=file_name
        )
        
        # 构建响应数据
        response_data = {
            "md_content": md_content,
            "file_name": file_name,
            "backend": backend,
            "pages_processed": f"{start_page}-{end_page if end_page else 'end'}"
        }
        
        # 添加可选的返回数据
        if return_middle_json:
            response_data["middle_json"] = middle_json
        
        if return_model_output:
            if isinstance(infer_result, list):
                model_output = "\n" + "-" * 50 + "\n".join(infer_result)
            else:
                model_output = str(infer_result)
            response_data["model_output"] = model_output
        
        if return_images:
            image_paths = glob(f"{image_dir}/*.jpg") + glob(f"{image_dir}/*.png")
            if image_paths:
                response_data["images"] = {
                    os.path.basename(image_path): f"data:image/jpeg;base64,{encode_image(image_path)}"
                    for image_path in image_paths
                }
        
        # 保存文件到磁盘（如果请求）
        if save_files:
            output_path = os.path.join(output_dir, file_name)
            os.makedirs(output_path, exist_ok=True)
            
            # 保存MD文件
            with open(os.path.join(output_path, f"{file_name}.md"), "w", encoding="utf-8") as f:
                f.write(md_content)
            
            # 保存中间JSON
            with open(os.path.join(output_path, f"{file_name}_middle.json"), "w", encoding="utf-8") as f:
                json.dump(middle_json, f, ensure_ascii=False, indent=2)
            
            # 保存模型输出
            if isinstance(infer_result, list):
                model_output = "\n" + "-" * 50 + "\n".join(infer_result)
            else:
                model_output = str(infer_result)
            with open(os.path.join(output_path, f"{file_name}_model_output.txt"), "w", encoding="utf-8") as f:
                f.write(model_output)
            
            response_data["saved_path"] = output_path
        
        logger.info(f"Successfully processed file: {file_name} with backend: {backend}")
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during VLM processing: {e}")
        return JSONResponse(
            content={"error": f"处理文件时发生错误: {str(e)}"},
            status_code=500
        )


@app.post(
    "/batch_vlm_parse",
    tags=["批量VLM解析"],
    summary="批量使用VLM模型解析多个文件",
    description="批量处理多个PDF或图像文件"
)
async def batch_vlm_parse(
    files: List[UploadFile],
    backend: str = Form("vlm-transformers", description="VLM后端类型"),
    server_url: Optional[str] = Form(None, description="SGLang服务器URL"),
    start_page: int = Form(0, description="开始页码", ge=0),
    end_page: Optional[int] = Form(None, description="结束页码", ge=0),
    output_dir: str = Form("output", description="输出目录"),
    return_images: bool = Form(False, description="是否返回图像"),
    save_files: bool = Form(True, description="是否保存文件到磁盘")
):
    """批量处理多个文件"""
    if not files:
        raise HTTPException(status_code=400, detail="请提供至少一个文件")
    
    if len(files) > 10:  # 限制批量处理的文件数量
        raise HTTPException(status_code=400, detail="批量处理最多支持10个文件")
    
    results = []
    failed_files = []
    
    for file in files:
        try:
            file_bytes = await file.read()
            file_name = Path(file.filename).stem if file.filename else "uploaded_file"
            file_extension = Path(file.filename).suffix.lower() if file.filename else ""
            
            if not file_extension:
                failed_files.append({"file": file.filename, "error": "无法确定文件类型"})
                continue
            
            # 处理单个文件
            middle_json, infer_result, md_content, image_dir = process_vlm_file(
                file_bytes=file_bytes,
                file_extension=file_extension,
                backend=backend,
                server_url=server_url,
                start_page_id=start_page,
                end_page_id=end_page,
                output_dir=output_dir,
                file_name=file_name
            )
            
            result = {
                "file_name": file_name,
                "status": "success",
                "md_content": md_content
            }
            
            if return_images:
                image_paths = glob(f"{image_dir}/*.jpg") + glob(f"{image_dir}/*.png")
                if image_paths:
                    result["images"] = {
                        os.path.basename(image_path): f"data:image/jpeg;base64,{encode_image(image_path)}"
                        for image_path in image_paths
                    }
            
            # 保存文件
            if save_files:
                output_path = os.path.join(output_dir, file_name)
                os.makedirs(output_path, exist_ok=True)
                
                with open(os.path.join(output_path, f"{file_name}.md"), "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                result["saved_path"] = output_path
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}")
            failed_files.append({"file": file.filename, "error": str(e)})
    
    response_data = {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(failed_files),
        "results": results
    }
    
    if failed_files:
        response_data["failed_files"] = failed_files
    
    return JSONResponse(content=response_data, status_code=200)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    ) 