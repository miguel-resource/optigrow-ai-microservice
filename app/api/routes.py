"""
Endpoints de la API para interactuar con Google Gemini
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    GenerateRequest, 
    GenerateResponse, 
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerateVideoFromImagesRequest,
    GenerateVideoFromImagesResponse,
    DownloadVideoRequest,
    ErrorResponse, 
    HealthResponse
)
from app.services.gemini_service import GeminiService
from app.api.auth import verify_api_key
from config.settings import settings
import logging
import io

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Endpoint para verificar el estado del servicio
    """
    temp_service = GeminiService(api_key=settings.gemini_api_key)
    
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        provider="Google Gemini",
        models_available=temp_service.get_available_models()
    )


@router.post(
    "/generate-text",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def generate_text(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Genera texto usando Google Gemini
    
    - **prompt**: Texto de entrada para el modelo
    - **model**: Modelo específico de Gemini (gemini-2.5-flash, gemini-1.5-flash, etc.)
    - **max_tokens**: Número máximo de tokens a generar (opcional)
    - **temperature**: Temperatura para la generación (0.0 a 2.0)
    - **top_p**: Top-p sampling (opcional)
    - **top_k**: Top-k sampling (opcional)
    """
    try:
        logger.info(f"Solicitud de generación con modelo: {request.model}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        result = await service.generate_text(
            prompt=request.prompt,
            model_name=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        logger.info(f"Generación exitosa con modelo: {request.model}")
        
        return GenerateResponse(
            success=True,
            model=request.model,
            text=result["text"],
            usage=result.get("usage")
        )
        
    except Exception as e:
        logger.error(f"Error al generar texto: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar texto: {str(e)}"
        )


@router.post(
    "/generate-video",
    response_model=GenerateVideoResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def generate_video(
    request: GenerateVideoRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Genera un video usando el modelo Veo de Gemini
    
    - **prompt**: Descripción de texto del video a generar
    - **model**: Modelo Veo a utilizar (veo-3.1-generate-preview, etc.)
    - **reference_images**: Lista de imágenes de referencia (máximo 3, URLs o base64)
    - **first_frame**: Imagen para el primer fotograma
    - **last_frame**: Imagen para el último fotograma
    - **aspect_ratio**: Relación de aspecto (16:9 o 9:16)
    - **resolution**: Resolución del video (720p o 1080p)
    - **duration_seconds**: Duración en segundos (4, 6 u 8)
    - **negative_prompt**: Elementos que no se quieren en el video
    """
    try:
        logger.info(f"Solicitud de generación de video con modelo: {request.model}")
        logger.info(f"Parámetros: duración={request.duration_seconds}s, resolución={request.resolution}, aspecto={request.aspect_ratio}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        # Procesar imágenes de referencia si se proporcionan
        reference_images = None
        if request.reference_images:
            # Aquí procesarías las imágenes (URLs o base64)
            # Por simplicidad, asumimos que ya están en el formato correcto
            reference_images = request.reference_images
        
        # Procesar primera y última imagen si se proporcionan
        first_frame = request.first_frame
        last_frame = request.last_frame
        
        result = await service.generate_video(
            prompt=request.prompt,
            model_name=request.model,
            # Simplificar para testing inicial - comentar parámetros complejos
            # reference_images=reference_images,
            # first_frame=first_frame,
            # last_frame=last_frame,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            duration_seconds=request.duration_seconds,
            # negative_prompt=request.negative_prompt
        )
        
        logger.info(f"Video generado exitosamente con modelo: {request.model}")
        
        # Si se solicitó descarga directa, retornar el archivo de video
        if request.download_directly:
            logger.info("Descargando video directamente...")
            video_file = service.get_video_from_cache(result["operation_id"])
            
            if video_file is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error al obtener video del caché"
                )
            
            video_data = await service.download_video(
                video_file=video_file,
                filename=f"video_{result['operation_id']}.mp4"
            )
            
            logger.info(f"Video descargado directamente ({len(video_data)} bytes)")
            
            return StreamingResponse(
                io.BytesIO(video_data),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=generated_video.mp4",
                    "Content-Length": str(len(video_data))
                }
            )
        
        # No incluir video_uri en la respuesta ya que no puede ser accedido directamente
        # El usuario debe usar download_url que maneja la autenticación
        return GenerateVideoResponse(
            success=True,
            model=request.model,
            video_uri=None,  # URI interna, no accesible directamente
            operation_id=result["operation_id"],
            duration_seconds=result["duration_seconds"],
            resolution=result["resolution"],
            aspect_ratio=result["aspect_ratio"],
            usage=result.get("usage"),
            download_url=f"http://localhost:8000/api/v1/download-video/{result['operation_id']}"
        )
        
    except Exception as e:
        logger.error(f"Error al generar video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar video: {str(e)}"
        )


@router.post(
    "/download-video",
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Video file"},
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def download_video(
    request: DownloadVideoRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Descarga un video generado por Veo
    
    - **operation_id**: ID de la operación de generación de video
    
    Retorna el archivo de video como descarga directa
    """
    try:
        logger.info(f"Solicitud de descarga de video para operación: {request.operation_id}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        # Obtener el video del caché
        video_file = service.get_video_from_cache(request.operation_id)
        
        if video_file is None:
            logger.error(f"Video no encontrado en caché para operation_id: {request.operation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video no encontrado. El video puede haber expirado o el operation_id es incorrecto."
            )
        
        # Descargar el video REAL usando la API de Gemini
        video_data = await service.download_video(
            video_file=video_file,
            filename=f"video_{request.operation_id}.mp4"
        )
        
        logger.info(f"Video REAL descargado para operación: {request.operation_id} ({len(video_data)} bytes)")
        
        # Retornar como respuesta de streaming
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=generated_video_{request.operation_id}.mp4",
                "Content-Length": str(len(video_data))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al descargar video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al descargar video: {str(e)}"

@router.post(
    "/generate-video-from-images",
    response_model=GenerateVideoFromImagesResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def generate_video_from_images(
    request: GenerateVideoFromImagesRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Genera un video cinematográfico a partir de una secuencia de imágenes
    
    - **images**: Lista de imágenes (URLs o base64, mínimo 2, máximo 10)
    - **prompt**: Descripción narrativa que conecta las imágenes
    - **model**: Modelo Veo a utilizar
    - **transition_style**: Estilo de transición (smooth, crossfade, morph, zoom, slide)
    - **aspect_ratio**: Relación de aspecto (16:9, 9:16, 1:1)
    - **resolution**: Resolución (720p, 1080p)
    - **duration_seconds**: Duración total (4-30 segundos)
    - **fps**: Fotogramas por segundo (24, 30, 60)
    - **interpolation_frames**: Frames de interpolación (6-24)
    - **motion_strength**: Intensidad del movimiento (0.0-1.0)
    - **zoom_effect**: Aplicar efecto zoom
    - **pan_direction**: Dirección de paneo (left, right, up, down)
    - **fade_transitions**: Usar fundidos suaves
    - **style**: Estilo cinematográfico (cinematic, documentary, artistic)
    - **seed**: Semilla para reproducibilidad
    - **download_directly**: Si retorna el archivo directamente
    """
    try:
        logger.info(f"Solicitud de generación de video desde {len(request.images)} imágenes")
        logger.info(f"Estilo: {request.transition_style}, Duración: {request.duration_seconds}s, FPS: {request.fps}")
        logger.info(f"Prompt: {request.prompt[:100]}...")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        result = await service.generate_video_from_images(
            images=request.images,
            prompt=request.prompt,
            model_name=request.model,
            transition_style=request.transition_style,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            interpolation_frames=request.interpolation_frames,
            motion_strength=request.motion_strength,
            zoom_effect=request.zoom_effect,
            pan_direction=request.pan_direction,
            fade_transitions=request.fade_transitions,
            style=request.style,
            seed=request.seed
        )
        
        logger.info(f"Video desde imágenes generado exitosamente: {result['operation_id']}")
        logger.info(f"Metadata: {result['image_count']} imágenes, {len(result['transitions'])} transiciones")
        
        # Si se solicitó descarga directa, retornar el archivo de video
        if request.download_directly:
            logger.info("Descargando video desde imágenes directamente...")
            video_file = service.get_video_from_cache(result["operation_id"])
            
            if video_file is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error al obtener video del caché"
                )
            
            video_data = await service.download_video(
                video_file=video_file,
                filename=f"video_from_images_{result['operation_id']}.mp4"
            )
            
            logger.info(f"Video desde imágenes descargado directamente ({len(video_data)} bytes)")
            
            return StreamingResponse(
                io.BytesIO(video_data),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=video_from_images_{result['operation_id']}.mp4",
                    "Content-Length": str(len(video_data))
                }
            )
        
        # Retornar metadata del video generado
        return GenerateVideoFromImagesResponse(
            success=True,
            model=request.model,
            video_uri=None,  # URI interna, no accesible directamente
            operation_id=result["operation_id"],
            metadata=result["metadata"],
            image_count=result["image_count"],
            transitions=result["transitions"],
            usage=result.get("usage"),
            download_url=f"http://localhost:8000/api/v1/download-video/{result['operation_id']}",
            message=result["message"]
        )
        
    except ValueError as ve:
        logger.error(f"Error de validación en generate_video_from_images: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error de validación: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error al generar video desde imágenes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar video desde imágenes: {str(e)}"
        )        )


@router.get(
    "/download-video/{operation_id:path}",
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Video file"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def download_video_get(operation_id: str):
    """
    Descarga un video generado por Veo usando GET (sin autenticación para facilitar acceso)
    
    - **operation_id**: ID de la operación de generación de video
    
    Retorna el archivo de video como descarga directa
    """
    try:
        logger.info(f"Descarga GET de video para operación: {operation_id}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        # Obtener el video del caché
        video_file = service.get_video_from_cache(operation_id)
        
        if video_file is None:
            logger.error(f"Video no encontrado en caché para operation_id: {operation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video no encontrado. El video puede haber expirado o el operation_id es incorrecto."
            )
        
        # Descargar el video REAL usando la API de Gemini
        video_data = await service.download_video(
            video_file=video_file,
            filename=f"video_{operation_id}.mp4"
        )
        
        logger.info(f"Video REAL descargado para operación: {operation_id} ({len(video_data)} bytes)")
        
        # Retornar como respuesta de streaming
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=generated_video_{operation_id}.mp4",
                "Content-Length": str(len(video_data))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al descargar video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al descargar video: {str(e)}"
        )


@router.get(
    "/models",
    response_model=dict,
    tags=["AI Models"]
)
async def list_models():
    """
    Lista todos los modelos de Gemini disponibles para generación de texto
    """
    service = GeminiService(api_key=settings.gemini_api_key)
    return {
        "success": True,
        "provider": "Google Gemini",
        "models": service.get_available_models()
    }
