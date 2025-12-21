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
        
        return GenerateVideoResponse(
            success=True,
            model=request.model,
            video_uri=result.get("video_uri"),
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
        
        # Obtener la operación por ID usando la API real
        from google.genai import types
        operation = types.GenerateVideosOperation(name=request.operation_id)
        operation = service.client.operations.get(operation)
        
        if not operation.done:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La generación del video aún no está completa"
            )
        
        if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontró video generado para esta operación"
            )
        
        # Obtener el video
        video_file = operation.response.generated_videos[0].video
        
        # Descargar el video usando la API real
        video_data = await service.download_video(video_file)
        
        logger.info(f"Video real descargado para operación: {request.operation_id}")
        
        # Retornar como respuesta de streaming
        return StreamingResponse(
            io.BytesIO(fake_video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=generated_video_{operation_id}.mp4",
                "Content-Length": str(len(fake_video_data)),
                "X-Video-Duration": "8",
                "X-Video-Info": f"Simulated 8-second video - Operation: {operation_id}"
            }
        )
        
        # TODO: Implementar cuando Veo esté disponible
        # # Obtener la operación por ID
        # from google.genai import types
        # operation = types.GenerateVideosOperation(name=request.operation_id)
        # operation = service.client.operations.get(operation)
        # 
        # if not operation.done:
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="La generación del video aún no está completa"
        #     )
        # 
        # if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
        #     raise HTTPException(
        #         status_code=status.HTTP_404_NOT_FOUND,
        #         detail="No se encontró video generado para esta operación"
        #     )
        # 
        # # Obtener el video
        # video_file = operation.response.generated_videos[0].video
        # 
        # # Descargar el video
        # video_data = await service.download_video(video_file)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al descargar video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al descargar video: {str(e)}"
        )


@router.get(
    "/download-video/{operation_id}",
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
        
        # Obtener la operación por ID usando la API real
        from google.genai import types
        operation = types.GenerateVideosOperation(name=operation_id)
        operation = service.client.operations.get(operation)
        
        if not operation.done:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La generación del video aún no está completa"
            )
        
        if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No se encontró video generado para esta operación"
            )
        
        # Obtener el video
        video_file = operation.response.generated_videos[0].video
        
        # Descargar el video usando la API real
        video_data = await service.download_video(video_file)
        
        logger.info(f"Video real descargado para operación: {operation_id}")
        
        # Retornar como respuesta de streaming
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=generated_video_{operation_id}.mp4",
                "Content-Length": str(len(video_data))
            }
        )
        
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
