"""
Modelos de datos para las peticiones y respuestas de la API
"""
from pydantic import BaseModel, Field, field_validator
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
    download_directly: bool = Field(
        default=False,
        description="Si es True, retorna el archivo de video directamente en lugar de metadata"
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
    video_uri: Optional[str] = Field(
        default=None, 
        description="URI interna del video (solo para referencia, no accesible directamente)"
    )
    operation_id: str = Field(..., description="ID de la operación de generación")
    duration_seconds: int = Field(..., description="Duración del video en segundos")
    resolution: str = Field(..., description="Resolución del video")
    aspect_ratio: str = Field(..., description="Relación de aspecto del video")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Información sobre el uso de tokens")
    download_url: str = Field(..., description="URL para descargar el video (usar esta URL)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model": "veo-3.1-generate-preview",
                "video_uri": None,
                "operation_id": "models/veo-3.1-generate-preview/operations/abc123",
                "duration_seconds": 4,
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "usage": {
                    "prompt_tokens": 15,
                    "video_tokens": 1000,
                    "total_tokens": 1015
                },
                "download_url": "http://localhost:8000/api/v1/download-video/models/veo-3.1-generate-preview/operations/abc123"
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


class GenerateVideoFromImagesRequest(BaseModel):
    """Modelo de petición para generación de video a partir de imágenes"""
    images: List[str] = Field(
        ..., 
        description="Lista de imágenes (URLs o base64), mínimo 2, máximo 10",
        min_length=2,
        max_length=10
    )
    prompt: str = Field(
        ..., 
        description="Descripción narrativa que conecta las imágenes y define el estilo del video"
    )
    model: str = Field(
        default="veo-3.1-generate-preview", 
        description="Modelo Veo a utilizar",
        examples=["veo-3.1-generate-preview", "veo-3.1-fast-preview"]
    )
    transition_style: str = Field(
        default="smooth", 
        description="Estilo de transición entre imágenes",
        pattern="^(smooth|crossfade|morph|zoom|slide)$"
    )
    aspect_ratio: str = Field(
        default="9:16", 
        description="Relación de aspecto del video",
        pattern="^(16:9|9:16|1:1)$"
    )
    resolution: str = Field(
        default="1080p", 
        description="Resolución del video",
        pattern="^(720p|1080p)$"
    )
    duration_seconds: int = Field(
        default=15, 
        description="Duración total del video en segundos (8, 15, 22, 29 o 58)",
        ge=8, le=58
    )
    fps: int = Field(
        default=30, 
        description="Fotogramas por segundo",
        ge=24, le=60
    )
    interpolation_frames: int = Field(
        default=8, 
        description="Frames de interpolación entre imágenes",
        ge=6, le=24
    )
    motion_strength: float = Field(
        default=0.2, 
        description="Intensidad del movimiento aplicado",
        ge=0.0, le=1.0
    )
    zoom_effect: bool = Field(
        default=False, 
        description="Aplicar efecto de zoom sutil en cada imagen"
    )
    pan_direction: Optional[str] = Field(
        default=None, 
        description="Dirección de paneo (opcional)"
    )
    fade_transitions: bool = Field(
        default=True, 
        description="Usar fundidos suaves entre transiciones"
    )
    style: Optional[str] = Field(
        default="natural_reel", 
        description="Estilo cinematográfico"
    )
    seed: Optional[int] = Field(
        default=None, 
        description="Semilla para reproducibilidad"
    )
    download_directly: bool = Field(
        default=False,
        description="Si es True, retorna el archivo de video directamente en lugar de metadata"
    )
    # Campos adicionales opcionales para compatibilidad con Laravel
    is_product_showcase: Optional[bool] = Field(
        default=True,
        description="Si es un showcase de producto"
    )
    maintain_context: Optional[bool] = Field(
        default=True,
        description="Mantener coherencia entre segmentos"
    )
    add_narration: Optional[bool] = Field(
        default=True,
        description="Incluir narración continua"
    )
    text_overlays: Optional[bool] = Field(
        default=True,
        description="Agregar texto overlay dinámico"
    )
    text_overlay_level: Optional[str] = Field(
        default="minimal",
        description="Nivel de texto dinámico: 'none', 'minimal', 'moderate', 'extensive'",
        pattern="^(none|minimal|moderate|extensive)$"
    )
    dynamic_camera_changes: Optional[bool] = Field(
        default=False,
        description="Usar cambios de cámara en extensiones"
    )
    audio_sync: Optional[bool] = Field(
        default=True,
        description="Sincronizar con audio mejorado"
    )
    use_nano_banana: Optional[bool] = Field(
        default=True,
        description="Usar Nano Banana para coherencia visual"
    )
    
    @field_validator('duration_seconds', mode='before')
    @classmethod
    def convert_duration_to_int(cls, v):
        """Convierte string a int para duration_seconds"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"duration_seconds debe ser un número entero, recibido: {v}")
        return v
    
    @field_validator('fps', mode='before') 
    @classmethod
    def convert_fps_to_int(cls, v):
        """Convierte string a int para fps"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"fps debe ser un número entero, recibido: {v}")
        return v
    
    @field_validator('interpolation_frames', mode='before')
    @classmethod
    def convert_interpolation_frames_to_int(cls, v):
        """Convierte string a int para interpolation_frames"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"interpolation_frames debe ser un número entero, recibido: {v}")
        return v
    
    @field_validator('motion_strength', mode='before')
    @classmethod
    def convert_motion_strength_to_float(cls, v):
        """Convierte string a float para motion_strength"""
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"motion_strength debe ser un número decimal, recibido: {v}")
        return v
    
    @field_validator('fps')
    @classmethod
    def validate_fps(cls, v):
        if v not in [24, 30, 60]:
            raise ValueError('fps debe ser 24, 30 o 60')
        return v
    
    @field_validator('duration_seconds')
    @classmethod
    def validate_duration_seconds(cls, v):
        if v not in [8, 15, 22, 29, 58]:
            raise ValueError('duration_seconds debe ser 8, 15, 22, 29 o 58')
        return v
    
    @field_validator('pan_direction')
    @classmethod
    def validate_pan_direction(cls, v):
        if v is not None and v not in ['left', 'right', 'up', 'down']:
            raise ValueError('pan_direction debe ser left, right, up o down')
        return v
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        if v is not None and v not in ['cinematic', 'documentary', 'artistic', 'natural_reel', 'reel_optimized']:
            raise ValueError('style debe ser cinematic, documentary, artistic, natural_reel o reel_optimized')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "images": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                    "https://example.com/image3.jpg"
                ],
                "prompt": "Un viaje mágico a través de bosques encantados al atardecer",
                "model": "veo-3.1-generate-preview",
                "transition_style": "smooth",
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "duration_seconds": 15,
                "fps": 24,
                "motion_strength": 0.8,
                "zoom_effect": False,
                "pan_direction": "right",
                "fade_transitions": True,
                "style": "cinematic"
            }
        }


class GenerateVideoFromImagesResponse(BaseModel):
    """Modelo de respuesta para generación de video desde imágenes"""
    success: bool = Field(..., description="Indica si la petición fue exitosa")
    model: str = Field(..., description="Modelo utilizado")
    video_uri: Optional[str] = Field(
        default=None, 
        description="URI interna del video (solo para referencia, no accesible directamente)"
    )
    operation_id: str = Field(..., description="ID de la operación de generación")
    metadata: Dict[str, Any] = Field(..., description="Metadata detallada del video generado")
    image_count: int = Field(..., description="Número de imágenes procesadas")
    transitions: List[Dict[str, Any]] = Field(..., description="Lista de transiciones aplicadas")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Estadísticas de uso de tokens")
    download_url: str = Field(..., description="URL para descargar el video (usar esta URL)")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model": "veo-3.1-generate-preview",
                "video_uri": None,
                "operation_id": "models/veo-3.1-generate-preview/operations/xyz789",
                "metadata": {
                    "source_images": 3,
                    "duration_seconds": 15,
                    "fps": 24,
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "transition_style": "smooth",
                    "total_transitions": 2
                },
                "image_count": 3,
                "transitions": [
                    {
                        "from_image": 1,
                        "to_image": 2,
                        "style": "smooth",
                        "duration": 0.5,
                        "timestamp": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "image_processing_tokens": 150,
                    "total_tokens": 425
                },
                "download_url": "http://localhost:8000/api/v1/download-video/xyz789",
                "message": "Video cinematográfico generado exitosamente desde 3 imágenes con transiciones smooth"
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
