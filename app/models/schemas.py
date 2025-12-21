"""
Modelos de datos para las peticiones y respuestas de la API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateRequest(BaseModel):
    """Modelo de petición para generación de texto"""
    prompt: str = Field(..., description="Texto de entrada para el modelo")
    model: str = Field(
        default="gemini-2.0-flash-exp", 
        description="Modelo específico de Gemini a utilizar",
        examples=["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
    )
    max_tokens: Optional[int] = Field(default=None, description="Número máximo de tokens a generar")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura para la generación")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "¿Cuáles son los mejores consejos para cultivar tomates?",
                "model": "gemini-2.5-flash",
                "temperature": 0.7
            }
        }


class GenerateResponse(BaseModel):
    """Modelo de respuesta para generación de texto"""
    success: bool = Field(..., description="Indica si la petición fue exitosa")
    model: str = Field(..., description="Modelo utilizado")
    text: str = Field(..., description="Texto generado por el modelo")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Información sobre el uso de tokens")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model": "gemini",
                "text": "Aquí están los mejores consejos para cultivar tomates...",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60
                }
            }
        }


class ErrorResponse(BaseModel):
    """Modelo de respuesta para errores"""
    success: bool = Field(default=False, description="Siempre False en errores")
    error: str = Field(..., description="Mensaje de error")
    detail: Optional[str] = Field(default=None, description="Detalles adicionales del error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid API key",
                "detail": "The provided API key is not valid"
            }
        }


class GenerateVideoRequest(BaseModel):
    """Modelo de petición para generación de video"""
    prompt: str = Field(..., description="Descripción de texto del video a generar")
    model: str = Field(
        default="veo-3.1-generate-preview", 
        description="Modelo Veo a utilizar",
        examples=["veo-3.1-generate-preview", "veo-3.1-fast-preview"]
    )
    reference_images: Optional[List[str]] = Field(
        default=None, 
        description="Lista de URLs o base64 de imágenes de referencia (máximo 3)"
    )
    first_frame: Optional[str] = Field(
        default=None, 
        description="Imagen para el primer fotograma (URL o base64)"
    )
    last_frame: Optional[str] = Field(
        default=None, 
        description="Imagen para el último fotograma (URL o base64)"
    )
    aspect_ratio: str = Field(
        default="16:9", 
        description="Relación de aspecto del video",
        pattern="^(16:9|9:16)$"
    )
    resolution: str = Field(
        default="720p", 
        description="Resolución del video",
        pattern="^(720p|1080p)$"
    )
    duration_seconds: int = Field(
        default=4, 
        description="Duración en segundos",
        ge=1, le=10
    )
    negative_prompt: Optional[str] = Field(
        default=None, 
        description="Elementos que no se quieren en el video"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Un gato jugando en un jardín soleado, cinematográfico, 4K",
                "model": "veo-3.1-generate-preview",
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "duration_seconds": 4,
                "negative_prompt": "cartoon, drawing, low quality"
            }
        }


class GenerateVideoResponse(BaseModel):
    """Modelo de respuesta para generación de video"""
    success: bool = Field(..., description="Indica si la petición fue exitosa")
    model: str = Field(..., description="Modelo utilizado")
    video_uri: Optional[str] = Field(default=None, description="URI del video generado")
    operation_id: str = Field(..., description="ID de la operación de generación")
    duration_seconds: int = Field(..., description="Duración del video en segundos")
    resolution: str = Field(..., description="Resolución del video")
    aspect_ratio: str = Field(..., description="Relación de aspecto del video")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Información sobre el uso de tokens")
    download_url: Optional[str] = Field(default=None, description="URL directa para descargar el video")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model": "veo-3.1-generate-preview",
                "video_uri": "gs://bucket/video.mp4",
                "operation_id": "operation-12345",
                "duration_seconds": 4,
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "usage": {
                    "prompt_tokens": 15,
                    "video_tokens": 1000,
                    "total_tokens": 1015
                },
                "download_url": "http://localhost:8000/api/v1/download-video/operation-12345"
            }
        }


class HealthResponse(BaseModel):
    """Modelo de respuesta para el endpoint de salud"""
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    provider: str = Field(..., description="Proveedor de IA")
    models_available: List[str] = Field(..., description="Lista de modelos de Gemini disponibles")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "provider": "Google Gemini",
                "models_available": [
                    "gemini-2.0-flash-exp",
                    "gemini-2.5-flash",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "veo-3.1-generate-preview"
                ]
            }
        }


class DownloadVideoRequest(BaseModel):
    """Modelo de petición para descargar video"""
    operation_id: str = Field(..., description="ID de la operación de generación de video")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation_id": "operation-12345"
            }
        }
