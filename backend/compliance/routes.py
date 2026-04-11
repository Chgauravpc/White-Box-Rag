import os
import logging
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from pydantic import BaseModel

from compliance.brd_parser import parse_brd
from compliance.mapper import map_requirement
from compliance.audit import get_all_logs, get_audit_by_id
from shared.models import BRDRequirement

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Compliance & Audit"])

class ValidateRequest(BaseModel):
    requirements: List[str]

@router.post("/brd/upload")
async def upload_brd(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Uploaded BRD is empty"})

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid file type. Only PDF, DOCX, and TXT are supported."})
    
    file_size = 0
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)

    try:
        async with aiofiles.open(temp_filepath, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > 10 * 1024 * 1024:
                    out_file.close()
                    os.remove(temp_filepath)
                    return JSONResponse(status_code=400, content={"status": "error", "message": "File too large. Max size is 10MB."})
                await out_file.write(chunk)
                
        if file_size == 0:
            os.remove(temp_filepath)
            return JSONResponse(status_code=400, content={"status": "error", "message": "Uploaded BRD is empty"})
            
        parsed_reqs = await parse_brd(temp_filepath)
        
        if not parsed_reqs:
            raise ValueError("No requirements extracted. The file might be unreadable.")
            
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        return {
            "status": "success",
            "message": "BRD parsed successfully",
            "data": [r.model_dump() for r in parsed_reqs]
        }
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return JSONResponse(status_code=500, content={"status": "error", "message": "Parsing failed", "details": str(e)})

@router.post("/brd/validate")
async def validate_brd(request: ValidateRequest):
    if not request.requirements:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing or empty requirements list."})
        
    try:
        results = []
        for req_text in request.requirements:
            logger.info(f"Validating requirement: {req_text}")
            brd_req = BRDRequirement(id="N/A", text=req_text)
            mapped_result = await map_requirement(brd_req)
            results.append(mapped_result)
            
        return {
            "status": "success",
            "message": "Validation complete",
            "data": results
        }
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to validate requirements", "details": str(e)})

@router.get("/brd/sample")
async def get_sample_brd():
    """Utility endpoint to fetch realistic demo requirements."""
    try:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_brd.txt")
        if not os.path.exists(sample_path):
            return JSONResponse(status_code=404, content={"status": "error", "message": "Sample BRD not found"})
            
        async with aiofiles.open(sample_path, 'r') as f:
            content = await f.read()
        return {
            "status": "success",
            "message": "Sample BRD loaded",
            "data": {"txt": content}
        }
    except Exception as e:
        logger.error(f"Error fetching sample BRD: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to get sample", "details": str(e)})

@router.get("/audit/logs")
async def list_audit_logs():
    try:
        logs = get_all_logs()
        return {
            "status": "success",
            "message": "Audit logs retrieved",
            "data": logs
        }
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to retrieve audit logs"})

@router.get("/audit/{id}")
async def get_audit_report(id: int):
    try:
        report = get_audit_by_id(id)
        if not report:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Audit report not found."})
            
        return {
            "status": "success",
            "message": "Report found",
            "data": report
        }
    except Exception as e:
        logger.error(f"Error getting audit report {id}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to retrieve audit report"})

@router.get("/audit/{id}/download")
async def download_audit_report(id: int):
    try:
        report = get_audit_by_id(id)
        if not report:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Audit report not found."})
            
        return JSONResponse(
            content=report,
            headers={"Content-Disposition": f'attachment; filename="audit_report_{id}.json"'}
        )
    except Exception as e:
        logger.error(f"Error downloading audit report {id}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to download audit report"})
