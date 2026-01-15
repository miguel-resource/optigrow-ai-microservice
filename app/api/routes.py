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
        
        reference_images = None
        if request.reference_images:
            reference_images = request.reference_images
        
        first_frame = request.first_frame
        last_frame = request.last_frame
        
        result = await service.generate_video(
            prompt=request.prompt,
            model_name=request.model,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            duration_seconds=request.duration_seconds
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
        
        return GenerateVideoResponse(
            success=True,
            model=request.model,
            video_uri=None,
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
        
        video_file = service.get_video_from_cache(request.operation_id)
        
        if video_file is None:
            logger.error(f"Video no encontrado en caché para operation_id: {request.operation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video no encontrado. El video puede haber expirado o el operation_id es incorrecto."
            )
        
        video_data = await service.download_video(
            video_file=video_file,
            filename=f"video_{request.operation_id}.mp4"
        )
        
        logger.info(f"Video REAL descargado para operación: {request.operation_id} ({len(video_data)} bytes)")
        
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
        )

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
    - **duration_seconds**: Duración total (15-57 segundos)
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
        )


@router.get(
    "/video-preview/{operation_id:path}",
    responses={
        200: {"content": {"application/json": {}}, "description": "Video preview URL"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def get_video_preview(operation_id: str):
    """
    Obtiene la URL de previsualización de un video generado por Veo
    
    - **operation_id**: ID de la operación de generación de video
    
    Retorna la URL para visualizar el video en el navegador
    """
    try:
        logger.info(f"Obteniendo preview de video para operación: {operation_id}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        video_file = service.get_video_from_cache(operation_id)
        
        if video_file is None:
            logger.error(f"Video no encontrado en caché para operation_id: {operation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video no encontrado. El video puede haber expirado o el operation_id es incorrecto."
            )
        
        # Obtener la URI del video para previsualización
        video_uri = None
        
        # Intentar obtener URI del objeto video_file
        try:
            if hasattr(video_file, 'uri'):
                uri_value = getattr(video_file, 'uri')
                if uri_value and isinstance(uri_value, str):
                    video_uri = uri_value
                    logger.info(f"URI obtenida de video_file.uri: {video_uri}")
        except Exception as e:
            logger.warning(f"Error accediendo a video_file.uri: {e}")
        
        if not video_uri:
            for attr_name in ['url', 'download_url', '_uri', 'path', 'file_uri']:
                try:
                    if hasattr(video_file, attr_name):
                        attr_value = getattr(video_file, attr_name)
                        if attr_value and isinstance(attr_value, str) and 'http' in attr_value:
                            video_uri = attr_value
                            logger.info(f"URI encontrada en {attr_name}: {video_uri}")
                            break
                except Exception as e:
                    continue
        
        if not video_uri:
            logger.error(f"No se pudo obtener URI para el video: {operation_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No se pudo obtener la URL de previsualización del video"
            )
        
        proxy_url = f"{settings.api_base_url}/api/v1/stream-video/{operation_id}"
        
        logger.info(f"Video preview URL generada para operación: {operation_id}")
        
        return {
            "success": True,
            "operation_id": operation_id,
            "video_uri": proxy_url,
            "original_uri": video_uri,
            "message": "URL de previsualización generada exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener preview de video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener preview de video: {str(e)}"
        )


@router.get(
    "/stream-video/{operation_id:path}",
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Video file download"},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["AI Models"]
)
async def stream_video(operation_id: str):
    """
    Descarga el video como archivo MP4
    Actúa como proxy entre el cliente y Google Cloud Storage
    
    - **operation_id**: ID de la operación de generación de video
    """
    try:
        logger.info(f"Descargando video para operación: {operation_id}")
        
        service = GeminiService(api_key=settings.gemini_api_key)
        
        video_file = service.get_video_from_cache(operation_id)
        
        if video_file is None:
            logger.info(f"Video no en caché, intentando recuperar de API: {operation_id}")
            video_file = await service.get_video_by_operation_id(operation_id)
        
        if video_file is None:
            logger.error(f"Video no encontrado después de búsqueda exhaustiva: {operation_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video no encontrado para operation_id: {operation_id}. Verifica que la operación haya completado exitosamente."
            )
        
        # Intentar obtener la URI del video de múltiples formas
        video_uri = None
        video_data = None
        
        for uri_attr in ['uri', 'url', 'download_url', '_uri', 'file_uri', 'location']:
            if hasattr(video_file, uri_attr):
                uri_value = getattr(video_file, uri_attr)
                if uri_value and isinstance(uri_value, str) and ('http' in uri_value or 'gs://' in uri_value):
                    video_uri = uri_value
                    logger.info(f"URI obtenida de {uri_attr}: {video_uri}")
                    break
        
        if not video_uri:
            for data_attr in ['video_bytes', '_content', 'data', 'content']:
                if hasattr(video_file, data_attr):
                    data_value = getattr(video_file, data_attr)
                    if data_value and len(data_value) > 0:
                        video_data = data_value
                        logger.info(f"Datos de video obtenidos directamente de {data_attr}: {len(video_data)} bytes")
                        break
        
        if video_uri:
            logger.info(f"Descargando video desde URI: {video_uri}")
            
            import requests
            headers = {}
            if settings.gemini_api_key and 'googleapis.com' in video_uri:
                headers['x-goog-api-key'] = settings.gemini_api_key
            
            response = requests.get(video_uri, headers=headers, stream=True, timeout=300)
            response.raise_for_status()
            
            video_data = response.content
            logger.info(f"Video descargado desde URI: {len(video_data)} bytes")
        
        if not video_data:
            logger.error(f"No se pudo obtener URI ni datos directos del video. Atributos del objeto: {dir(video_file)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No se pudo obtener los datos del video. Atributos disponibles: {[attr for attr in dir(video_file) if not attr.startswith('_')]}"
            )
        
        safe_operation_id = operation_id.split('/')[-1] if '/' in operation_id else operation_id
        filename = f"video_{safe_operation_id}.mp4"
        
        return StreamingResponse(
            io.BytesIO(video_data),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(video_data)),
                "Content-Type": "video/mp4"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al hacer streaming del video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al hacer streaming del video: {str(e)}"
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


@router.get(
    "/video-cache",
    response_model=dict,
    tags=["Debug"]
)
async def list_video_cache():
    """
    Lista todos los videos en caché (solo para debugging)
    """
    from app.services.gemini_service import _video_cache
    
    cache_keys = list(_video_cache.keys())
    cache_details = {}
    
    for key in cache_keys:
        video_obj = _video_cache[key]
        cache_details[key] = {
            "type": str(type(video_obj)),
            "attributes": [attr for attr in dir(video_obj) if not attr.startswith('_')],
            "has_uri": hasattr(video_obj, 'uri'),
            "has_video_bytes": hasattr(video_obj, 'video_bytes'),
            "uri_value": getattr(video_obj, 'uri', None) if hasattr(video_obj, 'uri') else None
        }
    
    return {
        "success": True,
        "cache_count": len(cache_keys),
        "cached_operations": cache_keys,
        "cache_details": cache_details,
        "message": f"Hay {len(cache_keys)} videos en caché"
    }


@router.get(
    "/debug-video/{operation_id:path}",
    response_model=dict,
    tags=["Debug"]
)
async def debug_video_search(operation_id: str):
    """
    Busca un video específico con debugging detallado
    """
    from app.services.gemini_service import _video_cache
    
    service = GeminiService(api_key=settings.gemini_api_key)
    
    search_result = {
        "operation_id": operation_id,
        "cache_keys": list(_video_cache.keys()),
        "direct_match": operation_id in _video_cache,
        "search_variations": [],
        "partial_matches": [],
        "video_found": False,
        "video_details": None
    }
    
    variations = [
        operation_id,
        operation_id.replace("models/", "").replace("projects/", ""),
        operation_id.split("/")[-1] if "/" in operation_id else operation_id,
        operation_id.replace("projects/", "video_models/") if operation_id.startswith("projects/") else operation_id,
        operation_id.replace("models/", "video_models/") if operation_id.startswith("models/") else operation_id
    ]
    
    for variation in variations:
        found = variation in _video_cache
        search_result["search_variations"].append({
            "variation": variation,
            "found": found
        })
        if found and not search_result["video_found"]:
            search_result["video_found"] = True
            video_obj = _video_cache[variation]
            search_result["video_details"] = {
                "found_with_variation": variation,
                "type": str(type(video_obj)),
                "attributes": [attr for attr in dir(video_obj) if not attr.startswith('_')],
                "has_uri": hasattr(video_obj, 'uri'),
                "uri_value": getattr(video_obj, 'uri', None) if hasattr(video_obj, 'uri') else None
            }
    
    for cached_key in _video_cache.keys():
        operation_uuid = operation_id.split('/')[-1] if '/' in operation_id else operation_id
        cached_uuid = cached_key.split('/')[-1] if '/' in cached_key else cached_key
        
        if operation_uuid in cached_key or cached_uuid in operation_id or operation_uuid == cached_uuid:
            search_result["partial_matches"].append(cached_key)
            if not search_result["video_found"]:
                search_result["video_found"] = True
                video_obj = _video_cache[cached_key]
                search_result["video_details"] = {
                    "found_with_partial_match": cached_key,
                    "type": str(type(video_obj)),
                    "attributes": [attr for attr in dir(video_obj) if not attr.startswith('_')],
                    "has_uri": hasattr(video_obj, 'uri'),
                    "uri_value": getattr(video_obj, 'uri', None) if hasattr(video_obj, 'uri') else None
                }
    
    return {
        "success": True,
        "search_result": search_result
    }


@router.post(
    "/simulate-cache/{operation_id:path}",
    response_model=dict,
    tags=["Debug"]
)
async def simulate_cache_storage(operation_id: str):
    """
    Simula el almacenamiento en caché para debugging
    """
    from app.services.gemini_service import _video_cache
    
    class MockVideo:
        def __init__(self):
            self.uri = "https://example.com/fake-video.mp4"
            
        def __str__(self):
            return "Mock Video Object"
    
    mock_video = MockVideo()
    
    _video_cache[operation_id] = mock_video
    
    operation_uuid = operation_id.split('/')[-1] if '/' in operation_id else operation_id
    _video_cache[operation_uuid] = mock_video
    
    return {
        "success": True,
        "message": "Video simulado guardado en caché",
        "stored_keys": [operation_id, operation_uuid],
        "cache_size": len(_video_cache)
    }


@router.get(
    "/debug-disk-cache",
    response_model=dict,
    tags=["Debug"]
)
async def debug_disk_cache():
    """
    Verifica el caché persistido en disco
    """
    from app.services.gemini_service import load_cache_from_disk
    
    disk_cache = load_cache_from_disk()
    
    return {
        "success": True,
        "disk_cache_exists": len(disk_cache) > 0,
        "disk_cache_entries": list(disk_cache.keys()),
        "disk_cache_details": disk_cache,
        "message": f"Caché en disco contiene {len(disk_cache)} entradas"
    }

