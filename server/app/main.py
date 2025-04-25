import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, \
                    BackgroundTasks
from fastapi.responses import FileResponse

from app.detection import load_model, process_video

# Load the YOLO model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "yolov5s.pt")
model = load_model(MODEL_PATH)

app = FastAPI(
    title="Traffic Sign Shape Detection Service",
    version="0.1.0"
)

@app.post("/detect-video/", summary="Upload a video and get it back annotated")
async def detect_video(
    background_tasks: BackgroundTasks,              # <-- non-default first
    file: UploadFile = File(...,                    # <-- defaulted
        description="MP4/AVI/MOV video file"
    ),
    conf: float = Form(0.5,                         # <-- defaulted
        ge=0.0,
        le=1.0,
        description="Confidence threshold"
    )
):
    # Validate content type
    if file.content_type not in ("video/mp4", "video/avi", "video/mov"):
        raise HTTPException(415, "Unsupported video format")

    # Create a temp dir that we’ll clean up later
    tmp_dir = tempfile.mkdtemp()
    inp = os.path.join(tmp_dir, "input.mp4")
    outp = os.path.join(tmp_dir, "output.mp4")

    # Save upload
    with open(inp, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Run detection
    try:
        process_video(model, inp, outp, conf)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Processing failed: {e}")

    # Schedule cleanup after response is sent
    background_tasks.add_task(shutil.rmtree, tmp_dir, True)

    return FileResponse(
        outp,
        media_type="video/mp4",
        filename="annotated.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app",
                host="0.0.0.0",
                port=8000,
                reload=True)
