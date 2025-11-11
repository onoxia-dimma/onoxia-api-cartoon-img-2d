from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from utils.cartoon_gan_utils import convert_to_cartoon, save_cartoon
from config import BASE_URL, OUTPUT_DIR
import logging, io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI(
    title="OnóxIA Cartoon GAN API",
    description="Transforma imagens em cartoon 2D profissional usando Deep Learning",
    version="1.0"
)

app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

@app.post("/cartoon")
async def cartoon_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        cartoon_img = convert_to_cartoon(image_bytes)
        filename = save_cartoon(cartoon_img, OUTPUT_DIR)
        image_url = f"{BASE_URL}/images/{filename}"

        logging.info(f"Imagem cartoon criada: {image_url}")
        return JSONResponse({
            "status": "success",
            "image_url": image_url,
            "message": "Imagem transformada em cartoon com sucesso."
        })
    except Exception as e:
        logging.error(f"Erro ao processar imagem: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")
