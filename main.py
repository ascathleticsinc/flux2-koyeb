import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from diffusers import Flux2Pipeline
from huggingface_hub import login
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Flux.2 Image Generation Service")

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "diffusers/FLUX.2-dev-bnb-4bit")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = os.getenv("DEVICE", "cuda:0")
MAX_RESOLUTION = int(os.getenv("MAX_RESOLUTION", "1024"))

# Global pipeline
pipeline = None
model_ready = False

class GenerationRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: int = None

@app.on_event("startup")
async def load_model():
    global pipeline, model_ready
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        
        # Login to Hugging Face if token provided
        if HF_TOKEN:
            login(token=HF_TOKEN)
            logger.info("Logged in to Hugging Face")
        
        # Load quantized Flux.2 pipeline
        pipeline = Flux2Pipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Enable memory optimizations
        pipeline.enable_attention_slicing()
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        model_ready = True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    if model_ready:
        return {"status": "healthy", "model": MODEL_NAME, "device": DEVICE}
    return {"status": "loading"}, 503

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    try:
        # Validate resolution
        if request.width > MAX_RESOLUTION or request.height > MAX_RESOLUTION:
            raise HTTPException(
                status_code=400,
                detail=f"Resolution exceeds maximum of {MAX_RESOLUTION}px"
            )
        
        # Set seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        logger.info(f"Generating image: {request.prompt[:50]}...")
        
        # Generate image
        image = pipeline(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        logger.info("Image generated successfully")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Flux.2 Image Generation",
        "model": MODEL_NAME,
        "status": "ready" if model_ready else "loading",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
