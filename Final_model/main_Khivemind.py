from fastapi import FastAPI, UploadFile, File, HTTPException
from model import predict_audio_bytes

app = FastAPI(title="Bee vs Hornet Detection API")


@app.get("/")
def root():
    return {"message": "Bee/Hornet classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

        result = predict_audio_bytes(audio_bytes)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))