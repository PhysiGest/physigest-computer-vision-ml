from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from camera_capture import PiCameraCapture
from yolo_classifier import YOLOClassifier
import cv2
import io

app = FastAPI()

camera = PiCameraCapture()
classifier = YOLOClassifier(
    model_path="./yolov8n-cls.onnx", labels_path="./yolo-classes.txt"
)


@app.on_event("startup")
async def startup_event():
    camera.start_preview()


@app.on_event("shutdown")
async def shutdown_event():
    camera.stop_preview()


@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            img = camera.capture_image()
            _, buffer = cv2.imencode(".jpg", img)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/classify")
async def classify_image():
    try:
        img = camera.capture_image()
        class_name, confidence = classifier.classify(img)
        if class_name is None:
            raise HTTPException(status_code=500, detail="Classification failed")
        # Convert the confidence to a standard float
        confidence = float(confidence)
        return JSONResponse(
            content={"class_name": class_name, "confidence": confidence}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
