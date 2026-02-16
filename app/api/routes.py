from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from app.services.resume_parser import parse_resume_file
from app.services.recommender import recommend_jobs
from app.services.s3_service import upload_to_s3, list_resumes, delete_resume
from app.services.index_manager import reload_index_and_jobs
import os
import tempfile

router = APIRouter()


class DeleteRequest(BaseModel):
    key: str
    

@router.post("/recommend")
async def recommend(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        # Move S3 upload to background and delete file after
        background_tasks.add_task(upload_to_s3, tmp_path, file.filename, delete_after=True)
        
        resume_text = parse_resume_file(file)
        results = recommend_jobs(resume_text)

        return {
        "filename": file.filename,
        "no. of recommendations": len(results),
        "recommendations": results
        }
    finally:
        # We can't delete the file immediately because background task needs it
        # The background task should ideally handle deletion
        pass
        
@router.get("/resumes")
def get_all_resumes():
    return list_resumes()


@router.delete("/resumes")
def delete_resume_api(req: DeleteRequest):
    delete_resume(req.key)
    return {"status": "deleted"}


@router.post("/admin/reload-index")
def reload_index():
    reload_index_and_jobs()
    return {"status": "reloaded"}