from google import genai
from google.genai import types
from typing import Dict, Any, Optional, List, Union
from config.settings import settings
import requests
import os
import uuid
import subprocess
import shutil

import logging
import base64
import io
from PIL import Image
import requests
import time
import json
import tempfile
import datetime
from google.cloud import storage

# Importar funciones de optimización de prompts desde módulo separado
from app.services.prompt_optimizer import (
    enhance_prompt_consistency,
    detect_language,
    optimize_prompt_for_reel,
    get_text_overlay_specifications,
    validate_text_readability_prompt,
    get_narration_consistency_specs,
    build_reel_enhanced_prompt,
    build_showcase_enhanced_prompt,
    get_final_restrictions,
    build_extension_prompt
)

# Importar funciones de caché desde módulo separado
from app.services.cache_service import (
    get_video_cache,
    add_to_cache,
    get_from_cache,
    save_cache_to_disk,
    load_cache_from_disk,
    ConcatenatedVideo,
    LocalConcatenatedVideo
)

# Importar funciones de almacenamiento GCS desde módulo separado
from app.services.storage_service import (
    download_video_from_gcs,
    generate_signed_url,
    generate_gcs_output_uri,
    gcs_uri_to_https_url,
    upload_video_to_gcs,
    get_storage_client
)

# Importar funciones de procesamiento de video desde módulo separado
from app.services.video_processor import (
    check_ffmpeg_available,
    download_video_segment,
    concatenate_videos_ffmpeg,
    get_video_url_from_result,
    process_concatenated_video
)

logger = logging.getLogger(__name__)

# Referencia al caché global (ahora manejado por cache_service)
_video_cache = get_video_cache()

"""
GENERACIÓN DE VIDEOS CON VEO 3.1

Límites importantes de la API (ACTUALIZADOS Enero 2026):
- Veo tiene un límite estricto de 4, 6 u 8 segundos por generación base
- NUEVO LÍMITE MÁXIMO: 30 segundos totales (incluidas extensiones)
- CONCATENACIÓN: Videos de 58s mediante unión de 2 segmentos de 29s cada uno

Duraciones soportadas actualmente:
- Videos simples: 8, 15, 22, 29 segundos (generación directa)
- Videos concatenados: 58 segundos (2x 29s + FFmpeg automático)

Métodos clave:
- generate_video(): Generación base simple (4, 6 u 8s) - SIEMPRE FUNCIONA
- generate_video_from_images(): Para videos hasta 29s o concatenación automática para 58s
- _generate_concatenated_video(): Manejo interno de videos largos con FFmpeg
- extend_video(): Extiende un video existente - PUEDE NO ESTAR DISPONIBLE
"""


class GeminiService:
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Inicializa el servicio de Gemini
        
        Args:
            api_key: Clave API de Google Gemini
            model_name: Nombre del modelo a usar
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Usar el SDK estándar de Google AI en lugar de Vertex AI para acceso a generate_videos
        self.client = genai.Client(
            vertexai=settings.gemini_vertexai,
            project=settings.gemini_project_id,
            location=settings.gemini_location,
            http_options=types.HttpOptions(
                api_version="v1"
            )
        )
        logger.info(f"GeminiService inicializado con modelo: {model_name} (Google AI SDK)")
    
    def get_available_models(self) -> List[str]:
        """
        Retorna los modelos disponibles de Gemini
        
        Returns:
            Lista de nombres de modelos disponibles
        """
        return [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
    
    def _process_image_input(self, image_input: Union[str, Any]) -> Any:
        """
        Procesa una imagen de entrada que puede ser URL, base64 o objeto Image
        para usar con la API de Google AI
        
        Args:
            image_input: URL, string base64, o objeto de imagen
            
        Returns:
            PIL Image procesado que puede ser usado con Gemini
        """
        try:
            logger.info(f"Procesando imagen: tipo={type(image_input)}, tamaño={len(str(image_input)) if isinstance(image_input, str) else 'N/A'}")
            
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    logger.info("Procesando imagen desde URL")
                    response = requests.get(image_input)
                    response.raise_for_status()
                    image_bytes = response.content
                elif image_input.startswith('data:image'):
                    logger.info("Procesando imagen desde data URL base64")
                    header, data = image_input.split(',', 1)
                    logger.info(f"Header: {header}, Data length: {len(data)}")
                    image_bytes = base64.b64decode(data)
                    logger.info(f"Decoded image data length: {len(image_bytes)}")
                else:
                    logger.info("Procesando imagen desde base64 sin header")
                    image_bytes = base64.b64decode(image_input)
                    logger.info(f"Decoded image data length: {len(image_bytes)}")
                
                # Convertir bytes a objeto PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Image loaded: {pil_image.size}, format: {pil_image.format}")
                
                # Convertir a RGB si es necesario para consistencia
                if pil_image.mode not in ('RGB', 'RGBA'):
                    logger.info(f"Convirtiendo imagen de {pil_image.mode} a RGB")
                    pil_image = pil_image.convert('RGB')
                
                # Optimizar tamaño para Veo (mejor rendimiento con imágenes <= 1024px)
                max_size = 1024
                if max(pil_image.size) > max_size:
                    original_size = pil_image.size
                    pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Imagen redimensionada de {original_size} a {pil_image.size} para optimizar con Veo")
                
                logger.info("Imagen procesada exitosamente como PIL Image optimizado para Veo")
                return pil_image
                
            else:
                logger.info("Imagen ya es un objeto, verificando tipo")
                # Si es PIL Image, devolverlo tal como está
                if hasattr(image_input, 'save') and hasattr(image_input, 'size'):
                    logger.info("Ya es PIL Image, devolviendo sin modificar")
                    return image_input
                else:
                    logger.warning("Formato de imagen desconocido, intentando conversión")
                    return image_input
                
        except Exception as e:
            logger.error(f"Error al procesar imagen: {str(e)}")
            logger.error(f"Tipo de input: {type(image_input)}")
            if isinstance(image_input, str):
                logger.error(f"Primeros 100 caracteres: {image_input[:100]}")
            raise Exception(f"Error al procesar imagen: {str(e)}")
    
    async def generate_text(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            current_model = model_name if model_name else self.model_name
            
            config = {}
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            if temperature is not None:
                config["temperature"] = temperature
            if top_p is not None:
                config["top_p"] = top_p
            if top_k is not None:
                config["top_k"] = top_k
            
            response = self.client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=config if config else None
            )
            
            logger.info(f"Texto generado exitosamente ({len(response.text)} caracteres)")
            
            usage_metadata = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
                logger.info(f"Tokens consumidos: {usage_metadata['total_tokens']} (prompt: {usage_metadata['prompt_tokens']}, respuesta: {usage_metadata['completion_tokens']})")
            
            logger.info(f"Respuesta generada exitosamente. Longitud: {len(response.text)} caracteres")
            
            return {
                "text": response.text,
                "usage": usage_metadata
            }
            
        except Exception as e:
            logger.error(f"Error al generar con Gemini: {str(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"Detalles del error: {e.__dict__}")
            raise Exception(f"Error al generar con Gemini: {str(e)}")
    
    async def generate_video(
        self,
        prompt: str,
        model_name: Optional[str] = "veo-3.1-generate-preview",
        reference_images: Optional[List] = None,
        first_frame: Optional[Any] = None,
        last_frame: Optional[Any] = None,
        aspect_ratio: str = "9:16",
        resolution: str = "1080p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = "natural_reel",
        motion_strength: float = 0.2,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Genera un video usando el modelo Veo de Gemini con opciones mejoradas
        
        Args:
            prompt (str): Descripción detallada del video a generar. Se recomienda ser específico
                         sobre movimientos, escenas, y elementos visuales deseados.
            model_name (Optional[str]): Modelo Veo a utilizar (default: "veo-3.1-generate-preview")
                                      Otros modelos disponibles: "veo-3.0-generate", "veo-2.0-generate"
            reference_images (Optional[List]): Lista de imágenes de referencia (máximo 3 imágenes)
                                             para guiar el estilo visual del video
            first_frame (Optional[Any]): Imagen para el primer fotograma del video
            last_frame (Optional[Any]): Imagen para el último fotograma del video
            aspect_ratio (str): Relación de aspecto del video ("16:9", "9:16", "1:1") (default: "16:9")
            resolution (str): Resolución de salida ("720p", "1080p") (default: "720p")
            duration_seconds (int): Duración en segundos - SOLO valores base: 4, 6, 8 (default: 8)
                                  Nota: Para videos largos, usa generate_video_from_images que maneja bloques automáticamente
            negative_prompt (Optional[str]): Descripción de elementos que NO se desean en el video
            style (Optional[str]): Estilo visual específico ("cinematic", "realistic", "animation", "documentary")
            motion_strength (float): Intensidad del movimiento (0.0-1.0, default: 0.5)
            seed (Optional[int]): Semilla para reproducibilidad de resultados
        
        Returns:
            Dict[str, Any]: Diccionario con información del video generado incluyendo:
                - video: Objeto de video de Gemini
                - video_uri: URI del video generado
                - operation_id: ID único de la operación
                - metadata: Información sobre duración, resolución, etc.
                - usage: Estadísticas de uso de tokens
        
        Raises:
            Exception: Si hay errores en los parámetros, límites de API, o problemas de red
        """
        try:
            # Validación de parámetros mejorada
            if not prompt or len(prompt.strip()) < 10:
                raise ValueError("El prompt debe tener al menos 10 caracteres y ser descriptivo")
            
            valid_aspects = ["16:9", "9:16", "1:1"]
            if aspect_ratio not in valid_aspects:
                raise ValueError(f"aspect_ratio debe ser uno de: {valid_aspects}")
            
            valid_resolutions = ["720p", "1080p"]
            if resolution not in valid_resolutions:
                raise ValueError(f"resolution debe ser una de: {valid_resolutions}")
            
            # Veo tiene límite estricto de 4, 6 u 8 segundos por generación base
            valid_base_durations = [4, 6, 8]
            if duration_seconds not in valid_base_durations:
                raise ValueError(f"duration_seconds debe ser uno de: {valid_base_durations} (límite de Veo para generación base)")
            
            if motion_strength < 0.0 or motion_strength > 1.0:
                raise ValueError("motion_strength debe estar entre 0.0 y 1.0")
            
            logger.info(f"Iniciando generación de video: {duration_seconds}s, {resolution}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Main args - aspect_ratio: {aspect_ratio}, motion_strength: {motion_strength}, model: {model_name}")
            
            current_model = model_name if model_name else "veo-3.1-generate-preview"
            
            # Mejorar el prompt con contexto adicional
            enhanced_prompt = prompt
            if style:
                enhanced_prompt = f"[{style.upper()} style] {enhanced_prompt}"
            if negative_prompt:
                enhanced_prompt += f" | Evitar: {negative_prompt}"
            
            image = None
            if first_frame is None:
                logger.info("Generando imagen de referencia...")
                image_response = self.client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=prompt,
                    config={"response_modalities":['IMAGE']}
                )
                
                try:
                    image = image_response.parts[0].as_image()
                    logger.info("Imagen de referencia generada")
                except AttributeError:
                    # Fallback: extraer desde inline_data
                    if hasattr(image_response.parts[0], 'inline_data') and image_response.parts[0].inline_data:
                        import base64
                        image_bytes = base64.b64decode(image_response.parts[0].inline_data.data)
                        image = Image.open(io.BytesIO(image_bytes))
                        logger.info("Imagen extraída desde inline_data")
                    else:
                        logger.warning("No se pudo generar imagen de referencia")
                        image = None
            elif first_frame is not None:
                image = first_frame
            
            # Verificar disponibilidad del método
            if not hasattr(self.client.models, 'generate_videos'):
                logger.error("Método generate_videos no encontrado")
                raise Exception("generate_videos no disponible. Actualiza google-genai: pip install --upgrade google-genai")
            
            # Step 3: Generate video with Veo (exactamente como en la documentación)
            try:
                operation = self.client.models.generate_videos(
                    model=current_model,
                    prompt=enhanced_prompt,
                    image=image
                )
                logger.info(f"Operación iniciada: {operation.name}")
            except Exception as api_error:
                logger.error(f"Error en generate_videos: {str(api_error)}")
                raise Exception(f"Error en API de videos: {str(api_error)}")
            
            max_wait_time = 600
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Tiempo de espera agotado para la generación de video (>{max_wait_time}s)")
                
                if int(elapsed_time) % 30 == 0:  # Log cada 30 segundos
                    logger.info(f"Generando... ({int(elapsed_time)}s)")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            if not hasattr(operation, 'response') or not operation.response:
                raise Exception("La operación se completó pero no se obtuvo respuesta")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            generated_video = operation.response.generated_videos[0]
            
            # CÁLCULO REAL DEL CONSUMO basado en videos generados exitosamente
            # El SDK de google-genai para Veo no devuelve usage_metadata con tokens
            # Se calcula manualmente: videos_exitosos * duración_configurada
            
            # 1. Contar videos realmente generados (después de filtros de seguridad)
            videos_generated_count = len(operation.response.generated_videos)
            
            # 2. Calcular consumo real en segundos
            consumption_seconds = videos_generated_count * duration_seconds
            
            # 3. Estimar costo (aproximadamente $0.40 USD por segundo para Veo 3.1 Standard)
            cost_per_second = 0.40
            estimated_cost_usd = consumption_seconds * cost_per_second
            
            usage_metadata = {
                "videos_requested": 1,  # Siempre se pide 1 video
                "videos_generated": videos_generated_count,
                "duration_per_video_seconds": duration_seconds,
                "consumption_seconds": consumption_seconds,
                "estimated_cost_usd": estimated_cost_usd,
                "cost_per_second_usd": cost_per_second,
                "calculation_method": "real_count",
                "note": "Consumo calculado basado en videos exitosamente generados"
            }
            
            # Información adicional si hay usage_metadata en la respuesta (poco común en Veo)
            if hasattr(operation.response, 'usage_metadata') and operation.response.usage_metadata:
                usage_metadata.update({
                    "api_usage_metadata": {
                        "prompt_tokens": getattr(operation.response.usage_metadata, 'prompt_token_count', 0),
                        "video_tokens": getattr(operation.response.usage_metadata, 'video_token_count', 0),
                        "total_tokens": getattr(operation.response.usage_metadata, 'total_token_count', 0)
                    }
                })
                logger.info(f"API usage_metadata disponible: {usage_metadata['api_usage_metadata']}")
            
            logger.info(f"Consumo calculado: {consumption_seconds} segundos ({videos_generated_count} videos × {duration_seconds}s)")
            logger.info(f"Costo estimado: ${estimated_cost_usd:.2f} USD")
            
            # Advertir si algunos videos fueron filtrados por seguridad
            if videos_generated_count == 0:
                logger.warning("ADVERTENCIA: Ningún video fue generado. Posible filtro de seguridad (RAI). Consumo = 0.")
            elif videos_generated_count < 1:
                logger.warning(f"Algunos videos pueden haber sido filtrados por seguridad. Solo {videos_generated_count} de 1 generados.")
            
            logger.info(f"Video generado exitosamente: {duration_seconds}s, {resolution}")
            
            # Guardar en caché para descargas posteriores
            _video_cache[operation.name] = generated_video.video
            operation_uuid = operation.name.split('/')[-1] if '/' in operation.name else operation.name
            _video_cache[operation_uuid] = generated_video.video
            
            # Generar URL firmada si el video tiene URI de GCS
            video_uri = generated_video.video.uri if hasattr(generated_video.video, 'uri') else None
            signed_url = None
            
            if video_uri and video_uri.startswith('gs://'):
                try:
                    signed_url = generate_signed_url(video_uri, expiration_minutes=15)
                    logger.info(f"URL firmada generada para video: {signed_url[:50]}...")
                except Exception as e:
                    logger.warning(f"No se pudo generar URL firmada: {e}")
            
            return {
                "video": generated_video.video,
                "video_uri": video_uri,
                "signed_url": signed_url,
                "operation_id": operation.name,
                "duration_seconds": duration_seconds,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "usage": usage_metadata,
                "message": "Video generado exitosamente con Veo 3.1",
                "download_info": {
                    "gcs_uri": video_uri,
                    "public_url": signed_url,
                    "expires_in_minutes": 15 if signed_url else None
                }
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error al generar video con Veo: {error_message}")
            if hasattr(e, '__dict__'):
                logger.error(f"Detalles delgemini error: {e.__dict__}")
            
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                raise Exception("Límite de cuota excedido para Veo 3.1. El modelo está disponible pero se han agotado los créditos/límites de uso. Verifica tu plan en https://ai.dev/usage")
            elif "404" in error_message or "NOT_FOUND" in error_message:
                print("Modelo no encontrado el error message es:", error_message)
                raise Exception("Modelo Veo no encontrado. Puede no estar disponible en tu región o cuenta.")
            elif "403" in error_message or "PERMISSION_DENIED" in error_message:
                raise Exception("Acceso denegado al modelo Veo. Verifica que tu API key tenga permisos para usar modelos de video.")
            elif "401" in error_message or "UNAUTHENTICATED" in error_message:
                raise Exception("API key inválida o expirada.")
            else:
                raise Exception(f"Error de Veo API: {error_message}")
    
    async def extend_video(
        self,
        previous_video_obj: Any,
        prompt: str,
        extension_seconds: int = 7,
        model_name: str = "veo-3.1-generate-preview",
        aspect_ratio: str = "9:16"
    ) -> Dict[str, Any]:
        """
        Extiende un video existente por 7 segundos adicionales con cambios dinámicos de cámara.
        
        Args:
            previous_video_obj: Objeto de video de Gemini retornado por generación anterior
            prompt: Descripción de lo que sucede en la extensión (evolución de la historia)
            extension_seconds: Duración de la extensión (solo 7 segundos soportado en Veo 3.1)
            model_name: Modelo Veo a utilizar
            aspect_ratio: Debe coincidir con el video original
            dynamic_camera_changes: Aplicar cambios de toma y ángulos (default: True)
            
        Returns:
            Dict con información del video extendido
        """
        try:
            # Veo 3.1 solo soporta extensiones de 7 segundos
            valid_extension_durations = [7]
            if extension_seconds not in valid_extension_durations:
                raise ValueError(f"extension_seconds debe ser uno de: {valid_extension_durations}. Veo 3.1 cambió las reglas de extensión.")
            
            logger.info(f"Extendiendo video por {extension_seconds}s adicionales...")
            logger.info(f"Prompt extensión: {prompt}")
            
            # Para extensiones, necesitamos URI de salida en GCS
            output_storage_uri = None
            if settings.gcs_bucket_name:
                output_storage_uri = generate_gcs_output_uri(
                    bucket_name=settings.gcs_bucket_name,
                    base_path=settings.gcs_output_path
                )
                logger.info(f"Usando output_storage_uri para extensión: {output_storage_uri}")
            else:
                logger.warning("No se configuró GCS bucket, intentando sin storage URI...")

            # Configuración para extensiones que pueden requerir storage URI
            try:
                extension_config = types.GenerateVideosConfig(
                    duration_seconds=extension_seconds,
                    aspect_ratio=aspect_ratio
                )
                
                # Agregar output_storage_uri si está disponible
                if output_storage_uri:
                    extension_config.output_gcs_uri = output_storage_uri

                operation = self.client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    video=previous_video_obj,
                    config=extension_config
                )
                
            except Exception as config_error:
                logger.warning(f"Error con configuración de extensión: {config_error}")
                # Fallback: intentar sin storage URI
                extension_config = types.GenerateVideosConfig(
                    duration_seconds=extension_seconds
                )

                operation = self.client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    video=previous_video_obj,
                    config=extension_config
                )
            
            logger.info(f"Operación de extensión iniciada: {operation.name}")
            
            max_wait_time = 600
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Timeout en extensión de video (>{max_wait_time//60}min)")
                
                time.sleep(15)
                operation = self.client.operations.get(operation)
            
            # Log detallado de la respuesta para debugging
            logger.info(f"Operación de extensión completada. Has response: {hasattr(operation, 'response')}")
            if hasattr(operation, 'response') and operation.response:
                logger.info(f"Response type: {type(operation.response)}")
                logger.info(f"Response attributes: {dir(operation.response)}")
                logger.info(f"Has generated_videos: {hasattr(operation.response, 'generated_videos')}")
                
            # Verificar si hay errores específicos
            if hasattr(operation, 'error') and operation.error:
                error_detail = str(operation.error)
                logger.error(f"Operation error: {operation.error}")
                
                # Manejo específico de errores de Responsible AI
                if ("responsible ai" in error_detail.lower() or 
                    "sensitive words" in error_detail.lower() or
                    "violate google's responsible ai practices" in error_detail.lower()):
                    raise Exception(
                        f"Error de filtros de seguridad de Google AI. "
                        f"El prompt puede contener palabras sensibles. "
                        f"Error original: {error_detail}"
                    )
                
                # Error específico para videos grandes  
                elif "output storage uri is required" in error_detail.lower():
                    raise Exception(
                        "Video de extensión demasiado grande. Se requiere configurar un bucket de Google Cloud Storage. "
                        "Para videos largos (>8s), Vertex AI requiere almacenamiento en GCS. "
                        f"Error original: {error_detail}"
                    )
                else:
                    raise Exception(f"Error en extensión: {operation.error}")
            
            if not hasattr(operation, 'response') or not operation.response:
                # Intentar obtener más información del error
                error_detail = ""
                if hasattr(operation, 'metadata') and operation.metadata:
                    logger.error(f"Operation metadata: {operation.metadata}")
                    error_detail = f" Metadata: {operation.metadata}"
                raise Exception(f"Operación de extensión completada sin respuesta válida{error_detail}")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la extensión")
            
            generated_video = operation.response.generated_videos[0]
            
            # Guardar en caché
            _video_cache[operation.name] = generated_video.video
            operation_uuid = operation.name.split('/')[-1] if '/' in operation.name else operation.name
            _video_cache[operation_uuid] = generated_video.video
            
            # Generar URL firmada para la extensión
            video_uri = None
            signed_url = None
            
            # Verificar si el video se guardó en GCS
            if output_storage_uri and hasattr(generated_video.video, 'uri'):
                video_uri = generated_video.video.uri or output_storage_uri
            elif hasattr(generated_video.video, 'uri'):
                video_uri = generated_video.video.uri
            
            if video_uri and video_uri.startswith('gs://'):
                try:
                    signed_url = generate_signed_url(video_uri, expiration_minutes=15)
                    logger.info(f"URL firmada generada para extensión: {signed_url[:50]}...")
                except Exception as e:
                    logger.warning(f"No se pudo generar URL firmada para extensión: {e}")
            
            logger.info(f"Video extendido exitosamente: +{extension_seconds}s")
            
            return {
                "video": generated_video.video,
                "video_uri": video_uri,
                "signed_url": signed_url,
                "output_storage_uri": output_storage_uri,
                "operation_id": operation.name,
                "duration_seconds": extension_seconds,
                "metadata": {
                    "type": "extension",
                    "duration": extension_seconds,
                    "aspect_ratio": aspect_ratio
                },
                "download_info": {
                    "gcs_uri": video_uri,
                    "public_url": signed_url,
                    "expires_in_minutes": 15 if signed_url else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error en extensión de video: {str(e)}")
            raise Exception(f"Error en extensión de video: {str(e)}")
    
    async def generate_video_from_images(
        self,
        images: List[Union[str, Any]],
        prompt: str,
        model_name: Optional[str] = "veo-3.1-generate-preview",
        transition_style: str = "smooth",
        aspect_ratio: str = "9:16",
        resolution: str = "1080p",
        duration_seconds: int = 15,
        fps: int = 30,
        interpolation_frames: int = 8,
        motion_strength: float = 0.2,  # Reducido para menor movimiento
        zoom_effect: bool = False,
        pan_direction: Optional[str] = None,  # Sin paneo por defecto
        fade_transitions: bool = True,
        style: Optional[str] = "natural_reel",
        is_product_showcase: bool = True,
        maintain_context: bool = True,
        add_narration: bool = True,
        text_overlays: bool = True,
        dynamic_camera_changes: bool = False  # Deshabilitar por defecto
    ) -> Dict[str, Any]:
        """
        Genera un video cinematográfico a partir de una secuencia de imágenes
        
        Este método toma una colección de imágenes y las convierte en un video fluido
        aplicando transiciones, efectos de movimiento y sincronización temporal.
        
        Args:
            images (List[Union[str, Any]]): Lista de imágenes (URLs, base64, o objetos Image)
                                          Mínimo 2 imágenes, máximo 10 para mejor rendimiento
            prompt (str): Descripción narrativa que conecta las imágenes y define el estilo
                         del video. Ejemplo: "Un viaje épico a través de paisajes fantásticos"
            model_name (Optional[str]): Modelo Veo a utilizar (default: "veo-3.1-generate-preview")
            transition_style (str): Estilo de transición entre imágenes:
                                  - "smooth": Transición suave y natural
                                  - "crossfade": Fundido cruzado clásico
                                  - "morph": Morfeo gradual entre imágenes
                                  - "zoom": Acercamiento/alejamiento dinámico
                                  - "slide": Deslizamiento direccional
            aspect_ratio (str): Relación de aspecto ("16:9", "9:16", "1:1") (default: "16:9")
            resolution (str): Resolución de salida ("720p", "1080p") (default: "720p")
            duration_seconds (int): Duración total del video en segundos (default: 15)
                                  - Videos se generan con bloque base de 8s + extensiones de 7s
                                  - Valores válidos: 15, 22, 29, 36, 43, 50, 57
                                  - Nota: Videos largos toman más tiempo (~10-15 min por bloque/extensión)
            fps (int): Fotogramas por segundo (24, 30, 60) (default: 24)
            interpolation_frames (int): Frames de interpolación entre imágenes (6-16) (default: 8)
                                       Valores optimizados para reels: 6-10 mantienen calidad del producto
                                       Valores altos (12-16) pueden generar contenido interpolado no deseado.
            motion_strength (float): Intensidad del movimiento aplicado (0.0-1.0) (default: 0.4)
                                   Valores optimizados para reels: 0.3-0.5 mantienen coherencia del producto
                                   Valores altos (0.6-1.0) pueden generar contenido inconsistente
            zoom_effect (bool): Aplicar efecto de zoom sutil en cada imagen (default: False)
                              Deshabilitado por defecto para mejor coherencia en reels
            pan_direction (Optional[str]): Dirección de paneo optimizada para reels
                                         "subtle_left", "subtle_right", "gentle_up", "gentle_down", None
            fade_transitions (bool): Usar fundidos suaves entre transiciones (default: True)
            audio_sync (bool): Sincronizar con audio mejorado (default: True)
            style (Optional[str]): Estilo optimizado: "reel_optimized", "product_showcase", "social_media"
            seed (Optional[int]): Semilla para reproducibilidad
            use_nano_banana (bool): Usar Nano Banana para coherencia visual (default: True)
            is_product_showcase (bool): Optimizar para showcase de productos (default: True)
            maintain_context (bool): Mantener coherencia entre segmentos (default: True)
            add_narration (bool): Incluir narración continua y envolvente (default: True)
            text_overlays (bool): Agregar texto dynamic_camera_changesque aparece dinámicamente (default: True)
            dynamic_camera_changes (bool): Usar cambios de cámara en extensiones (default: True)
        
        Returns:
            Dict[str, Any]: Información completa del video generado:
                - video: Objeto de video de Gemini
                - video_uri: URI del video para descarga
                - operation_id: Identificador único de la operación
                - metadata: Detalles técnicos (duración, fps, resolución, etc.)
                - image_count: Número de imágenes procesadas
                - transitions: Lista de transiciones aplicadas
                - usage: Estadísticas de tokens y recursos utilizados
        
        Raises:
            ValueError: Si los parámetros no son válidos
            Exception: Si hay errores en el procesamiento o la API
        
        Example:
            ```python
            images = ["image1.jpg", "image2.jpg", "image3.jpg"]
            prompt = "Un viaje mágico a través de bosques encantados al atardecer"
            result = await service.generate_video_from_images(
                images=images,
                prompt=prompt,
                transition_style="smooth",
                duration_seconds=10,
                motion_strength=0.8
            )
            ```
        """
        try:
            logger.info(f"Iniciando generación de video estilo reel desde {len(images)} imágenes")
            
            # Detectar y ajustar contenido médico para evitar problemas de políticas
            medical_keywords = ['medical', 'medicina', 'medico', 'salud', 'health', 'tratamiento', 'treatment', 'terapia', 'therapy', 'dispositivo médico', 'medical device', 'enfermedad', 'disease', 'diagnóstico', 'diagnosis']
            is_medical_content = any(keyword.lower() in prompt.lower() for keyword in medical_keywords)
            
            # Inicializar optimized_prompt con el prompt original
            optimized_prompt = prompt
            
            if is_medical_content:
                logger.info("Contenido médico detectado. Ajustando prompt para cumplir políticas...")
                # Hacer el prompt más neutro y enfocado en características técnicas
                optimized_prompt = optimized_prompt.replace('medicina', 'tecnología')
                optimized_prompt = optimized_prompt.replace('medical', 'technical')
                optimized_prompt = optimized_prompt.replace('salud', 'bienestar')
                optimized_prompt = optimized_prompt.replace('health', 'wellness')
                optimized_prompt = optimized_prompt.replace('tratamiento', 'uso')
                optimized_prompt = optimized_prompt.replace('treatment', 'usage')
                optimized_prompt = f"Presentación profesional de producto tecnológico con características técnicas avanzadas"
                logger.info(f"Prompt ajustado para contenido médico: {optimized_prompt[:100]}...")
            
            # Optimizar prompt para estética de reel
            if maintain_context and not is_medical_content:  # Solo optimizar si no es contenido médico
                # Detectar idioma del prompt para mantener consistencia
                detected_language = detect_language(prompt)
                optimized_prompt = optimize_prompt_for_reel(optimized_prompt, is_product_showcase, add_narration, text_overlays, detected_language)
                logger.info(f"Prompt optimizado para coherencia, narración y estética de reel con idioma: {detected_language}")
            elif is_medical_content:
                detected_language = "spanish"
                logger.info("Prompt médico mantenido neutro para cumplir políticas")
            else:
                optimized_prompt = prompt
                detected_language = detect_language(prompt)
            
            # Validación exhaustiva de parámetros
            if not images or len(images) < 2:
                raise ValueError("Se requieren al menos 2 imágenes para generar un video")
            
            if len(images) > 10:
                logger.warning(f"Se proporcionaron {len(images)} imágenes, usando solo las primeras 10")
                images = images[:10]
            
            if not prompt or len(prompt.strip()) < 15:
                raise ValueError("El prompt debe ser descriptivo (mínimo 15 caracteres)")
            
            valid_transitions = ["smooth", "crossfade", "morph", "zoom", "slide"]
            if transition_style not in valid_transitions:
                raise ValueError(f"transition_style debe ser uno de: {valid_transitions}")
            
            valid_aspects = ["16:9", "9:16", "1:1"]
            if aspect_ratio not in valid_aspects:
                raise ValueError(f"aspect_ratio debe ser uno de: {valid_aspects}")
            
            # Ajustar duración para contenido médico (evitar extensiones problemáticas)
            if is_medical_content and duration_seconds > 8:
                logger.info("Contenido médico detectado - limitando a 8s para evitar problemas de políticas")
                duration_seconds = 8
                
            # Duraciones totales permitidas (ACTUALIZADO: máximo 30s + concatenación para 58s)
            # Base: 8s, Extensiones: 7s cada una
            # Simples: 8 (solo base), 15 (8+7), 22 (8+7+7), 29 (8+7+7+7)
            # Concatenados: 58 (29s + 29s unidos automáticamente)
            valid_total_durations = [8, 15, 22, 29, 58]
            if duration_seconds not in valid_total_durations:
                raise ValueError(f"duration_seconds debe ser uno de: {valid_total_durations} (58s = concatenación automática)")
            
            valid_fps = [24, 30, 60]
            if fps not in valid_fps:
                raise ValueError(f"fps debe ser uno de: {valid_fps}")
            
            if interpolation_frames < 6 or interpolation_frames > 24:
                raise ValueError("interpolation_frames debe estar entre 6 y 24")
            
            valid_pan_directions = ["subtle_left", "subtle_right", "gentle_up", "gentle_down", None]
            if pan_direction not in valid_pan_directions:
                raise ValueError(f"pan_direction debe ser uno de: {valid_pan_directions[:-1]} o None")
            
            logger.info(f"Generando video desde {len(images)} imágenes - {duration_seconds}s")
            
            # Detectar si necesitamos concatenación para videos largos (58s)
            if duration_seconds == 58:
                logger.info("Video de 58s detectado - usando concatenación de 2 segmentos de 29s")
                return await self._generate_concatenated_video(
                    images, optimized_prompt, 29, transition_style, aspect_ratio, 
                    motion_strength, fps, interpolation_frames, pan_direction, model_name,
                    is_product_showcase, maintain_context, add_narration, text_overlays, dynamic_camera_changes
                )
            
            # Determinar si necesitamos generar por bloques (para videos <= 29s)
            base_duration = 8  # Duración del bloque base
            extension_duration = 7  # Duración de cada extensión (Veo 3.1 spec)
            needs_extension = duration_seconds > base_duration
            num_extensions = 0
            
            if needs_extension:
                # Calcular cuántas extensiones necesitamos
                remaining_duration = duration_seconds - base_duration
                num_extensions = remaining_duration // extension_duration
                
                # Validación crítica: verificar que el total no exceda 30s (límite de Google Veo API)
                total_calculated = base_duration + (num_extensions * extension_duration)
                if total_calculated > 30:
                    raise ValueError(f"Duración calculada {total_calculated}s excede el límite máximo de 30s de Google Veo API. Usa una duración menor.")
                
                logger.info(f"Video largo detectado: {duration_seconds}s = {base_duration}s base + {num_extensions} extensiones de {extension_duration}s")
            
            # Procesar todas las imágenes
            processed_images = []
            for i, image in enumerate(images):
                try:
                    processed_img = self._process_image_input(image)
                    processed_images.append(processed_img)
                except Exception as e:
                    logger.error(f"Error procesando imagen {i+1}: {str(e)}")
                    raise ValueError(f"Error en imagen {i+1}: {str(e)}")
            
            logger.info(f"Imágenes procesadas: {len(processed_images)}")
            
            time_per_image = duration_seconds / len(processed_images)
            
            # Determinar si es contenido tipo reel basado en parámetros y configurar dinámicas de cámara
            is_reel_content = (style == "social_media" or aspect_ratio == "9:16" or 
                             duration_seconds >= 15 or "reel" in prompt.lower() or 
                             "tiktok" in prompt.lower() or "instagram" in prompt.lower() or
                             style == "reel_optimized")
            
            # Configurar movimiento de cámara basado en dynamic_camera_changes
            camera_movement = "dynamic" if dynamic_camera_changes else "static"
            logger.info(f"Configuración de cámara: {camera_movement}, Contenido reel: {is_reel_content}")
            
            if is_reel_content:
                # Prompt optimizado para contenido tipo reel/social media
                detected_language = "spanish" if any(word in optimized_prompt.lower() for word in ['producto', 'este', 'con', 'para', 'de', 'la', 'el']) else "english"
                
                enhanced_prompt = f"""
                REEL PROFESIONAL: Crear video {duration_seconds}s estilo Instagram/TikTok optimizado para móviles
                usando estas {len(processed_images)} imágenes del producto. {optimized_prompt}
                
                ESTILO REEL OPTIMIZADO:
                - Estética moderna y atractiva para redes sociales
                - Movimientos cinematográficos suaves y profesionales  
                - Iluminación y contraste optimizados para dispositivos móviles
                - Transiciones fluidas {transition_style} que mantengan engagement ({time_per_image:.1f}s por imagen)
                - Audio continuo y sincronizado sin cortes ni distorsión
                - Composición vertical perfecta para formato 9:16
                - CÁMARA DINÁMICA: {"Cambios de ángulo y perspectiva" if dynamic_camera_changes else "Movimientos suaves y constantes"}
                - NARRACIÓN CONTINUA EN {detected_language.upper()}: Voz envolvente que explique el producto durante todo el video
                - TEXTO DINÁMICO LEGIBLE EN {detected_language.upper()}: 
                  * Overlays con ALTO CONTRASTE (texto blanco sobre fondos oscuros o viceversa)
                  * Tamaño de fuente GRANDE y CLARO para móviles (mínimo 24pt equivalente)
                  * Tipografía BOLD y fácil de leer (Arial, Helvetica, sans-serif)
                  * Posición estratégica que no interfiera con el producto
                  * Duración suficiente para lectura (mínimo 3 segundos por texto)
                  * Animaciones suaves de entrada y salida
                - ELEMENTOS VISUALES: Textos destacados, títulos llamativos, información del producto
                
                COHERENCIA Y CONTINUIDAD:
                - MANTENER tema principal "{optimized_prompt}" en TODO el video
                - Audio limpio y continuo entre segmentos (sin cortes abruptos)
                - Cada segmento conecta naturalmente con el anterior
                - Preservar mensaje central y estilo visual consistente
                - Transiciones suaves que no rompan el flujo narrativo
                - IDIOMA CONSISTENTE: Solo {detected_language.upper()} durante todo el video
                - NARRACIÓN SIN REPETICIONES: Pronunciación clara y fluida
                - NO cambiar de idioma en ningún momento del video
                
                FIDELIDAD DEL PRODUCTO:
                - Mostrar EXACTAMENTE el producto de las imágenes proporcionadas
                - Mantener características, colores, forma y detalles específicos
                - NO inventar elementos que no estén en las imágenes originales
                - Aplicar efectos sin distorsionar la apariencia real del producto
                """
            else:
                # Prompt tradicional para videos no-reel optimizado para mejor audio y coherencia
                enhanced_prompt = f"""
                SHOWCASE PROFESIONAL: Crear video promocional de {duration_seconds} segundos estilo reel 
                usando estas {len(processed_images)} imágenes del producto. {optimized_prompt}
                
                AUDIO Y ESTÉTICA:
                - Audio continuo y sin cortes para mejor experiencia 
                - NARRACIÓN PROFESIONAL: Voz clara que describe el producto durante todo el video
                - TEXTO INFORMATIVO CLARO:
                  * Overlays con MÁXIMO CONTRASTE para perfecta legibilidad
                  * Fuente GRANDE y BOLD optimizada para dispositivos móviles
                  * Texto NÍTIDO sin pixelación ni borrosidad
                  * Posicionamiento que no obstruya el producto principal
                  * Información concisa y fácil de leer rápidamente
                  * Colores contrastantes (blanco/negro, amarillo/negro)
                - MOVIMIENTO DE CÁMARA: {"Dinámico con cambios de perspectiva" if dynamic_camera_changes else "Suave y consistente"}
                - Calidad profesional optimizada para redes sociales
                - Transiciones suaves que mantengan engagement
                - Movimientos cinematográficos controlados
                
                FIDELIDAD DEL PRODUCTO: 
                Mantener EXACTAMENTE las características, colores, forma y textura del producto real.
                NO inventar ni modificar elementos. Mostrar únicamente lo que aparece en las imágenes.
                
                Transiciones {transition_style} optimizadas para reel ({time_per_image:.1f}s por imagen).
                Audio sincronizado y de alta calidad.
                """
            
            if zoom_effect:
                enhanced_prompt += "\n- Aplicar zoom MUY sutil y controlado (máximo 10% de cambio) manteniendo EXACTAMENTE la integridad y apariencia del producto. NO inventar detalles durante el zoom."
            
            if pan_direction:
                camera_instruction = f"Movimiento de cámara {pan_direction} {'dinámico y variado' if dynamic_camera_changes else 'suave y profesional'}, optimizado para reel."
                enhanced_prompt += f"\n- {camera_instruction}"
            
            if fade_transitions:
                enhanced_prompt += "\n- Transiciones suaves y elegantes que mantengan el flujo del reel."
            
            if style and style != "reel_optimized":
                enhanced_prompt = f"[{style.upper()} REEL] {enhanced_prompt}"
            
            # Agregar especificaciones técnicas para texto legible
            if text_overlays:
                text_specs = get_text_overlay_specifications(aspect_ratio)
                enhanced_prompt += f"\n\n{text_specs}"
            
            # Agregar especificaciones de narración consistente
            if add_narration:
                detected_language = detect_language(optimized_prompt)
                narration_specs = get_narration_consistency_specs(detected_language)
                enhanced_prompt += f"\n\n{narration_specs}"
            
            # Agregar restricciones finales usando función del módulo de optimización
            enhanced_prompt += get_final_restrictions(dynamic_camera_changes)
            
            current_model = model_name if model_name else "veo-3.1-generate-preview"
            reference_image_obj = None
            
            if len(processed_images) > 0:
                first_image_pil = processed_images[0]
                
                if first_image_pil.mode not in ('RGB', 'RGBA'):
                    first_image_pil = first_image_pil.convert('RGB')
                
                reference_image_obj = first_image_pil
                
                enhanced_prompt += f"\n\nCrear video que muestre exactamente el producto específico de estas fotografías, "
                enhanced_prompt += f"manteniendo sus características auténticas y detalles específicos."
            else:
                raise ValueError("No se procesaron imágenes válidas")
            
            # Verificar que el método existe
            if not hasattr(self.client.models, 'generate_videos'):
                raise Exception("El método generate_videos no está disponible. Verifica la instalación del paquete google-genai")
            
            # Step 2: Generate video with Veo (usando PIL Image directamente)
            try:
                buffered = io.BytesIO()
                image_format = reference_image_obj.format if reference_image_obj.format else 'PNG'
                reference_image_obj.save(buffered, format=image_format)
                image_bytes = buffered.getvalue()

                image_input = types.Image(
                    image_bytes=image_bytes,
                    mime_type=f'image/{image_format.lower()}'
                )
                
                if not enhanced_prompt or not isinstance(enhanced_prompt, str):
                    final_prompt = "Professional product showcase with smooth movement, continuous narration, LARGE HIGH-CONTRAST text overlays, crystal clear typography, high quality, 4k, social media ready"
                else:
                    # Validar legibilidad del texto antes del envío
                    validated_prompt = validate_text_readability_prompt(enhanced_prompt, aspect_ratio)
                    final_prompt = validated_prompt
                    # Añadir elementos específicos para texto legible si no están presentes
                    if not any(word in final_prompt.lower() for word in ['legible', 'contrast', 'readable', 'clear']):
                        final_prompt += ", LARGE LEGIBLE text overlays with HIGH CONTRAST, readable typography, crystal clear text"

                # Configuración para videos, con manejo especial para videos largos
                import uuid
                output_filename = f"veo_video_{uuid.uuid4().hex[:8]}.mp4"
                
                # Usar configuración de GCS si está disponible
                output_storage_uri = None
                if settings.gcs_bucket_name:
                    output_storage_uri = generate_gcs_output_uri(
                        bucket_name=settings.gcs_bucket_name,
                        base_path=settings.gcs_output_path
                    )
                    logger.info(f"Usando GCS configurado: {output_storage_uri}")
                
                try:
                    generation_config = types.GenerateVideosConfig(
                        duration_seconds=base_duration,
                        aspect_ratio=aspect_ratio,
                        negative_prompt="low quality, distorted, deformed, ugly, blurry, grain, text, watermark, changed product, modified appearance, different product, altered features, fantasy elements, non-existent details, creative interpretation, artistic deviation from original"
                    )
                    
                    # Agregar output_storage_uri si está disponible (para videos que lo requieran)
                    if output_storage_uri:
                        generation_config.output_gcs_uri = output_storage_uri
                        logger.info(f"Configurado output_storage_uri: {output_storage_uri}")
                
                except Exception as config_error:
                    logger.warning(f"Error configurando GCS, usando configuración básica: {config_error}")
                    generation_config = types.GenerateVideosConfig(
                        duration_seconds=base_duration,
                        aspect_ratio=aspect_ratio,
                        negative_prompt="low quality, distorted, deformed, ugly, blurry, grain, text, watermark, changed product, modified appearance, different product, altered features, fantasy elements, non-existent details, creative interpretation, artistic deviation from original"
                    )

                operation = self.client.models.generate_videos(
                    model=current_model,
                    prompt=final_prompt,
                    image=image_input,
                    config=generation_config
                )
                
                logger.info(f"Operación iniciada: {operation.name}")
            except Exception as api_error:
                logger.error(f"Error en generate_videos: {str(api_error)}")
                
                # Verificar si es violación de políticas de Vertex AI
                if ("violates" in str(api_error).lower() and "usage guidelines" in str(api_error).lower()) or \
                   ("policy" in str(api_error).lower() and ("violation" in str(api_error).lower() or "guideline" in str(api_error).lower())):
                    
                    logger.warning("Violación de políticas detectada. Intentando con prompt sanitizado...")
                    
                    # Crear prompt completamente neutral sin menciones médicas
                    sanitized_prompt = f"""
                    Professional product demonstration video showcasing a technical consumer device. 
                    Clean presentation with smooth camera movements showing product features and design details.
                    High quality {aspect_ratio} format, {base_duration} seconds duration.
                    Neutral technical showcase without medical claims or health references.
                    Focus on: device design, build quality, technical specifications, user interface.
                    """
                    
                    try:
                        logger.info("Reintentando con prompt sanitizado...")
                        operation = self.client.models.generate_videos(
                            model=current_model,
                            prompt=sanitized_prompt,
                            image=image_input,
                            config=generation_config
                        )
                        logger.info("Generación exitosa con prompt sanitizado")
                    except Exception as sanitized_error:
                        logger.error(f"Error incluso con prompt sanitizado: {str(sanitized_error)}")
                        raise Exception(f"Vertex AI rechaza el contenido por políticas: {str(api_error)}. Incluso con prompt neutral falló: {str(sanitized_error)}")
                else:
                    raise Exception(f"Error en API de videos: {str(api_error)}")
            
            max_wait_time = 900
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Timeout en generación de video (>{max_wait_time//60}min)")
                
                time.sleep(15)
                operation = self.client.operations.get(operation)
            
            if not hasattr(operation, 'response') or not operation.response:
                # Verificar errores específicos y obtener más información de debugging
                error_detail = ""
                operation_status = "unknown"
                
                if hasattr(operation, 'error') and operation.error:
                    error_detail = str(operation.error)
                    logger.error(f"Operation error details: {error_detail}")
                    
                    if "output storage uri is required" in error_detail.lower():
                        raise Exception(
                            f"Video demasiado grande para generación directa. Se requiere bucket GCS configurado. "
                            f"Para videos de {duration_seconds}s, considera usar duraciones más cortas (4, 6 u 8 segundos) "
                            f"o configurar un bucket de Google Cloud Storage. Error: {error_detail}"
                        )
                    elif "safety" in error_detail.lower() or "responsible ai" in error_detail.lower():
                        raise Exception(
                            f"Video rechazado por filtros de seguridad (Responsible AI). "
                            f"Intenta modificar el prompt para ser menos específico sobre personas o marcas. "
                            f"Error: {error_detail}"
                        )
                
                # Obtener información adicional del estado de la operación
                if hasattr(operation, 'metadata'):
                    metadata_info = str(operation.metadata)
                    logger.error(f"Operation metadata: {metadata_info}")
                    error_detail += f" | Metadata: {metadata_info}"
                
                if hasattr(operation, 'done'):
                    operation_status = f"done={operation.done}"
                    logger.error(f"Operation status: {operation_status}")
                
                # Log completo para debugging
                logger.error(f"Operation attributes: {dir(operation)}")
                
                raise Exception(f"Operación completada sin respuesta válida. Status: {operation_status}. {error_detail}")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            generated_video = operation.response.generated_videos[0]
            current_video_obj = generated_video.video
            all_operation_ids = [operation.name]
            total_blocks = 1
            
            # Si necesitamos extender el video, hacemos extensiones sucesivas
            if needs_extension and num_extensions > 0:
                # Verificar configuración de GCS antes de intentar extensiones
                gcs_available = self.ensure_gcs_bucket()
                if not gcs_available:
                    logger.warning(f"Videos largos ({duration_seconds}s) requieren configuración de GCS.")
                    logger.info("Devolviendo video base de 8s sin extensiones.")
                    duration_seconds = base_duration  # Actualizar duración real
                    needs_extension = False
                    num_extensions = 0

                if needs_extension and num_extensions > 0:
                    logger.info(f"Iniciando {num_extensions} extensiones para alcanzar {duration_seconds}s totales")
                
                for ext_num in range(num_extensions):
                    logger.info(f"Generando extensión {ext_num + 1}/{num_extensions}...")
                    
                    # Prompt optimizado para la extensión manteniendo contexto reel con cambios dinámicos
                    detected_language = "spanish" if any(word in optimized_prompt.lower() for word in ['producto', 'este', 'con', 'para', 'de', 'la', 'el']) else "english"
                    
                    # Usar la función build_extension_prompt mejorada con contexto completo
                    core_theme = optimized_prompt.split('.')[0].strip()
                    
                    # Extraer características del narrador del prompt original para mantener consistencia
                    narrator_context = f"El narrador que presentó '{core_theme}' debe continuar con la misma voz y personalidad"
                    
                    extension_prompt = build_extension_prompt(
                        core_theme=f"{core_theme}. {narrator_context}",
                        detected_language=detected_language,
                        dynamic_camera_changes=False,  # Deshabilitar por defecto para consistencia
                        is_reel_content=is_reel_content
                    )
                    
                    try:
                        extension_result = await self.extend_video(
                            previous_video_obj=current_video_obj,
                            prompt=extension_prompt,
                            extension_seconds=7,
                            model_name=current_model,
                            aspect_ratio=aspect_ratio,
                            dynamic_camera_changes=dynamic_camera_changes
                        )
                        
                        # Actualizar el video actual para la próxima extensión
                        current_video_obj = extension_result["video"]
                        all_operation_ids.append(extension_result["operation_id"])
                        total_blocks += 1
                        
                        logger.info(f"Extensión {ext_num + 1} completada exitosamente")
                        
                    except Exception as ext_error:
                        error_message = str(ext_error)
                        logger.error(f"Error en extensión {ext_num + 1}: {error_message}")
                        
                        # Manejo específico de errores de Responsible AI
                        if ("responsible ai" in error_message.lower() or 
                            "sensitive words" in error_message.lower() or
                            "violate google's responsible ai practices" in error_message.lower()):
                            
                            logger.warning(f"Error de filtros de seguridad en extensión {ext_num + 1}. Intentando con prompt básico...")
                            
                            # Fallback a prompt ultra-simple
                            simple_prompt = "Continue video smoothly. Same style."
                            
                            try:
                                extension_result = await self.extend_video(
                                    previous_video_obj=current_video_obj,
                                    prompt=simple_prompt,
                                    extension_seconds=7,
                                    model_name=current_model,
                                    aspect_ratio=aspect_ratio
                                )
                                
                                current_video_obj = extension_result["video"]
                                all_operation_ids.append(extension_result["operation_id"])
                                total_blocks += 1
                                
                                logger.info(f"Extensión {ext_num + 1} completada con prompt fallback")
                                continue  # Continuar con la siguiente extensión
                                
                            except Exception as fallback_error:
                                logger.error(f"Fallback también falló en extensión {ext_num + 1}: {fallback_error}")
                                logger.warning(f"Terminando video en {base_duration + (ext_num * extension_duration)}s debido a errores")
                                break
                        
                        # Si el error es por falta de storage URI, ofrecer alternativa
                        elif "output storage uri is required" in error_message.lower():
                            logger.warning(f"Extensión {ext_num + 1} falló por requerir GCS. Devolviendo video base de {base_duration}s")
                            logger.info("Para videos más largos, configura un bucket de Google Cloud Storage")
                            # No lanzar error, usar el video base
                            break
                        elif "supported durations are [7]" in error_message.lower():
                            logger.error(f"Extensión {ext_num + 1} falló: Veo 3.1 requiere extensiones de 7s exactos")
                            logger.info("Intenta con duraciones válidas: 8, 15, 22, 29, 36, 43, 50, 57, 64s")
                            break
                        else:
                            raise Exception(f"Error en extensión {ext_num + 1}/{num_extensions}: {error_message}")
                
                # Si llegamos hasta aquí con extensiones exitosas
                if total_blocks > 1:
                    extension_total = (total_blocks - 1) * extension_duration
                    logger.info(f"Video completo generado en {total_blocks} bloques: {base_duration}s base + {total_blocks - 1} extensiones de {extension_duration}s = {base_duration + extension_total}s total")
                    # El video final es el último generado
                    generated_video.video = current_video_obj
                else:
                    logger.info(f"Video generado solo con bloque base: {base_duration}s (extensiones fallaron)")
                    duration_seconds = base_duration  # Actualizar duración real
            
            videos_generated_count = len(operation.response.generated_videos)
            
            consumption_seconds = videos_generated_count * duration_seconds
            
            # TODO: Valor actual del 2026, editar si este valor cambia
            cost_per_second = 0.40
            estimated_cost_usd = consumption_seconds * cost_per_second
            
            usage_metadata = {
                "videos_requested": 1,  # Siempre se pide 1 video completo
                "videos_generated": videos_generated_count,
                "duration_per_video_seconds": duration_seconds,
                "consumption_seconds": consumption_seconds,
                "estimated_cost_usd": estimated_cost_usd,
                "cost_per_second_usd": cost_per_second,
                "source_images_count": len(processed_images),
                "total_blocks": total_blocks if needs_extension else 1,
                "base_duration": base_duration if needs_extension else duration_seconds,
                "extensions_count": num_extensions if needs_extension else 0,
                "all_operation_ids": all_operation_ids,
                "calculation_method": "blocks" if needs_extension else "single",
                "note": f"Video generado en {total_blocks} bloque(s)" if needs_extension else "Video generado en un solo bloque"
            }
            
            if hasattr(operation.response, 'usage_metadata') and operation.response.usage_metadata:
                usage_metadata.update({
                    "api_usage_metadata": {
                        "prompt_tokens": getattr(operation.response.usage_metadata, 'prompt_token_count', 0),
                        "video_tokens": getattr(operation.response.usage_metadata, 'video_token_count', 0),
                        "total_tokens": getattr(operation.response.usage_metadata, 'total_token_count', 0)
                    }
                })
                logger.info(f"API usage_metadata disponible: {usage_metadata['api_usage_metadata']}")
            
            logger.info(f"Consumo calculado: {consumption_seconds} segundos ({videos_generated_count} videos × {duration_seconds}s)")
            logger.info(f"Costo estimado: ${estimated_cost_usd:.2f} USD")
            logger.info(f"Procesadas {len(processed_images)} imágenes fuente")
            
            if videos_generated_count == 0:
                logger.warning("ADVERTENCIA: Ningún video fue generado. Posible filtro de seguridad (RAI). Consumo = 0.")
            elif videos_generated_count < 1:
                logger.warning(f"Algunos videos pueden haber sido filtrados por seguridad. Solo {videos_generated_count} de 1 generados.")
            
            transitions_applied = []
            for i in range(len(processed_images) - 1):
                transitions_applied.append({
                    "from_image": i + 1,
                    "to_image": i + 2,
                    "style": transition_style,
                    "duration": interpolation_frames / fps,
                    "timestamp": i * time_per_image
                })
            
            metadata = {
                "source_images": len(processed_images),
                "duration_seconds": duration_seconds,
                "fps": fps,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "transition_style": transition_style,
                "interpolation_frames": interpolation_frames,
                "motion_strength": motion_strength,
                "zoom_effect": zoom_effect,
                "pan_direction": pan_direction,
                "fade_transitions": fade_transitions,
                "style": style,
                "time_per_image": time_per_image,
                "total_transitions": len(transitions_applied)
            }
            
            _video_cache[operation.name] = generated_video.video
            operation_uuid = operation.name.split('/')[-1] if '/' in operation.name else operation.name
            _video_cache[operation_uuid] = generated_video.video
            
            # Generar URL firmada para el video desde imágenes
            video_uri = generated_video.video.uri if hasattr(generated_video.video, 'uri') else None
            signed_url = None
            
            if video_uri and video_uri.startswith('gs://'):
                try:
                    signed_url = generate_signed_url(video_uri, expiration_minutes=15)
                    logger.info(f"URL firmada generada para video desde imágenes: {signed_url[:50]}...")
                except Exception as e:
                    logger.warning(f"No se pudo generar URL firmada para video desde imágenes: {e}")
            
            # Persistir caché para debugging
            save_cache_to_disk()
            
            logger.info(f"Video generado: {duration_seconds}s desde {len(processed_images)} imágenes")
            
            return {
                "video": generated_video.video,
                "video_uri": video_uri,
                "signed_url": signed_url,
                "operation_id": operation.name,
                "metadata": metadata,
                "image_count": len(processed_images),
                "transitions": transitions_applied,
                "usage": usage_metadata,
                "message": f"Video cinematográfico generado exitosamente desde {len(processed_images)} imágenes con transiciones {transition_style}",
                "download_info": {
                    "gcs_uri": video_uri,
                    "public_url": signed_url,
                    "expires_in_minutes": 15 if signed_url else None
                }
            }
            
        except ValueError as ve:
            logger.error(f"Error de validación en generate_video_from_images: {str(ve)}")
            raise ve
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error al generar video desde imágenes: {error_message}")
            
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                raise Exception("Límite de cuota excedido para Veo. Verifica tu plan en https://ai.dev/usage")
            elif "404" in error_message or "NOT_FOUND" in error_message:
                raise Exception("Modelo Veo no encontrado. Puede no estar disponible en tu región.")
            elif "403" in error_message or "PERMISSION_DENIED" in error_message:
                raise Exception("Acceso denegado al modelo Veo. Verifica permisos de tu API key.")
            elif "401" in error_message or "UNAUTHENTICATED" in error_message:
                raise Exception("API key inválida o expirada.")
            elif ("violates" in error_message.lower() and "usage guidelines" in error_message.lower()) or \
                 ("policy" in error_message.lower() and ("violation" in error_message.lower() or "guideline" in error_message.lower())):
                raise Exception("Contenido rechazado por políticas de Vertex AI. Las imágenes del producto pueden contener elementos que violan las directrices de uso. Intenta con imágenes más neutrales o descriptiones técnicas sin referencias médicas.")
            else:
                raise Exception(f"Error en generación de video desde imágenes: {error_message}")
    
    def get_video_from_cache(self, operation_id: str) -> Optional[Any]:
        """
        Obtiene un video del caché o directamente de la API por operation_id
        
        Args:
            operation_id: ID de la operación de generación
            
        Returns:
            Objeto de video de Gemini o None si no existe
        """
        # Buscar en caché
        logger.info(f"Buscando video en caché: {operation_id}")
        
        # Buscar directo en caché
        video = _video_cache.get(operation_id)
        if video:
            logger.info("Video encontrado en caché")
            return video
        
        # Intentar búsquedas alternativas
        search_variations = [
            operation_id.replace("models/", "").replace("projects/", ""),
            operation_id.split("/")[-1] if "/" in operation_id else operation_id,
        ]
        
        for variation in search_variations:
            video = _video_cache.get(variation)
            if video:
                logger.info(f"Video encontrado con variación: {variation}")
                _video_cache[operation_id] = video
                return video
        
        # Buscar por coincidencia parcial
        for cached_key in _video_cache.keys():
            operation_uuid = operation_id.split('/')[-1] if '/' in operation_id else operation_id
            cached_uuid = cached_key.split('/')[-1] if '/' in cached_key else cached_key
            
            if operation_uuid == cached_uuid:
                logger.info(f"Video encontrado por coincidencia: {cached_key}")
                video = _video_cache[cached_key]
                _video_cache[operation_id] = video
                return video
        
        logger.warning("Video no encontrado en caché")
        return None
    
    async def download_video(self, video_file, filename: str = "generated_video.mp4") -> bytes:
        """
        Descarga optimizada usando directamente el cliente de GCS
        evitando problemas de 401 en buckets privados.
        
        Args:
            video_file: Objeto de archivo de video de Gemini
            filename: Nombre del archivo de destino
            
        Returns:
            Bytes del video descargado
        """
        try:
            logger.info(f"Iniciando descarga nativa de GCS: {filename}")
            
            # 1. Obtener la URI
            video_uri = getattr(video_file, 'uri', None)
            if not video_uri:
                # Intentar buscar en otros atributos si 'uri' falla
                for attr in ['video_uri', 'file_uri', 'url']:
                    video_uri = getattr(video_file, attr, None)
                    if video_uri: break

            if not video_uri:
                raise ValueError("No se encontró URI en el objeto de video")

            # 2. Si es GCS, descargar usando el cliente nativo (sin requests.get)
            if video_uri.startswith('gs://'):
                # Parsear gs://bucket/blob_name
                path_parts = video_uri.replace("gs://", "").split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1]

                # Usar el cliente de Storage directamente
                # Esto usa tus credenciales actuales (aunque sean de usuario) 
                # y funcionará porque tienes permiso de lectura, sin necesitar firmar URL.
                storage_client = storage.Client(project=settings.gemini_project_id)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                logger.info(f"Descargando blob {blob_name} del bucket {bucket_name}...")
                video_bytes = blob.download_as_bytes()
                
                logger.info(f"Video descargado exitosamente ({len(video_bytes)} bytes)")
                return video_bytes

            else:
                # Fallback para URIs que no son GCS (poco probable con Veo)
                response = requests.get(video_uri, timeout=300)
                response.raise_for_status()
                logger.info(f"Video descargado desde URI no-GCS ({len(response.content)} bytes)")
                return response.content

        except Exception as e:
            logger.error(f"Error crítico descargando video: {str(e)}")
            raise Exception(f"Error al descargar video: {str(e)}")
        
    async def get_video_by_operation_id(self, operation_id: str) -> Any:
        """
        Recupera un video de la API de Gemini usando el operation_id
        
        Args:
            operation_id: ID de la operación de video (ej: models/veo-3.1-generate-preview/operations/abc123)
        
        Returns:
            Objeto de video de Gemini o None si no se encuentra
        """
        try:
            logger.info(f"Recuperando video de la API por operation_id: {operation_id}")
            
            # Intentar obtener el archivo directamente usando la API de Files
            # El operation_id contiene información del archivo generado
            
            # Extraer el file_id del operation_id si está disponible
            # El formato típico es: models/MODEL/operations/OPERATION_ID
            
            # Listar archivos recientes para encontrar el video
            try:
                files = genai.list_files()
                for file in files:
                    # Buscar por nombre o metadata que coincida con el operation_id
                    if hasattr(file, 'display_name') and operation_id in str(file.display_name):
                        logger.info(f"Video encontrado por display_name: {file.name}")
                        _video_cache[operation_id] = file
                        return file
                    
                    # También buscar en el nombre del archivo
                    if hasattr(file, 'name') and operation_id in str(file.name):
                        logger.info(f"Video encontrado por name: {file.name}")
                        _video_cache[operation_id] = file
                        return file
                        
                    # Buscar por coincidencia parcial en metadatos
                    file_metadata = str(getattr(file, 'metadata', ''))
                    if operation_id in file_metadata:
                        logger.info(f"Video encontrado por metadata: {file.name}")
                        _video_cache[operation_id] = file
                        return file
                        
            except Exception as e:
                logger.warning(f"No se pudo listar archivos: {e}")
            
            logger.warning(f"No se pudo recuperar video para operation_id: {operation_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error al recuperar video por operation_id: {str(e)}")
            return None
    
    def ensure_gcs_bucket(self) -> bool:
        """
        Verifica que existe configuración de GCS para almacenar videos grandes
        
        Returns:
            True si hay configuración de GCS disponible
        """
        try:
            if not settings.gcs_bucket_name:
                logger.warning("No se configuró GCS bucket. Videos largos pueden fallar.")
                logger.info("Para configurar GCS bucket, agrega en .env:")
                logger.info("GCS_BUCKET_NAME=tu-bucket-videos")
                logger.info("GCS_OUTPUT_PATH=videos/generated/")
                return False
            
            if not settings.gemini_project_id:
                logger.warning("No se configuró GEMINI_PROJECT_ID. URLs firmadas pueden fallar.")
                logger.info("Para configurar proyecto, agrega en .env:")
                logger.info("GEMINI_PROJECT_ID=tu-google-project-id")
                # No retornar False aquí, el bucket puede funcionar sin proyecto explícito
            
            # Verificar que el bucket sea válido
            try:
                if settings.gemini_project_id:
                    storage_client = storage.Client(project=settings.gemini_project_id)
                    logger.info(f"Verificando bucket con proyecto: {settings.gemini_project_id}")
                else:
                    storage_client = storage.Client()
                    logger.info("Verificando bucket sin proyecto explícito")
                    
                bucket = storage_client.bucket(settings.gcs_bucket_name)
                
                # Intenta hacer una operación simple para verificar acceso
                if not bucket.exists():
                    logger.warning(f"Bucket '{settings.gcs_bucket_name}' no existe.")
                    logger.info("Crea el bucket con: gsutil mb gs://tu-bucket-videos")
                    return False
                
                logger.info(f"GCS bucket configurado correctamente: {settings.gcs_bucket_name}")
                return True
                
            except Exception as bucket_error:
                logger.warning(f"Error accediendo al bucket GCS '{settings.gcs_bucket_name}': {bucket_error}")
                logger.info("Verifica que:")
                logger.info("1. El bucket existe en Google Cloud Storage")
                logger.info("2. Tienes permisos de acceso al bucket")
                logger.info("3. Las credenciales de Google Cloud están configuradas")
                logger.info("4. GEMINI_PROJECT_ID está configurado en .env")
                return False
                
        except Exception as e:
            logger.error(f"Error verificando configuración GCS: {str(e)}")
            return False

    async def _generate_concatenated_video(
        self,
        images: List[Any],
        base_prompt: str,
        segment_duration: int = 29,
        transition_style: str = "smooth",
        aspect_ratio: str = "9:16",
        motion_strength: float = 0.4,
        fps: int = 30,
        interpolation_frames: int = 8,
        pan_direction: Optional[str] = "subtle_left",
        model_name: str = "veo-3.1-generate-preview",
        is_product_showcase: bool = True,
        maintain_context: bool = True,
        add_narration: bool = True,
        text_overlays: bool = True,
        dynamic_camera_changes: bool = True
    ) -> Dict[str, Any]:
        """
        Genera un video largo (hasta 58s) concatenando múltiples segmentos con coherencia mejorada
        
        Optimizado para:
        - Estética de reel/Instagram
        - Coherencia visual entre segmentos  
        - Mejor manejo del audio
        - Transiciones suaves
        - Presentación profesional de productos
        """
        try:
            # Verificar que FFmpeg esté disponible
            check_ffmpeg_available()
            
            logger.info(f"Iniciando concatenación de video - 2 segmentos de {segment_duration}s")
            
            # Dividir imágenes en dos grupos para variedad
            mid_point = len(images) // 2
            images_segment1 = images[:mid_point] if mid_point > 0 else images
            images_segment2 = images[mid_point:] if mid_point > 0 else images
            
            # Asegurar que cada segmento tenga al menos 2 imágenes
            if len(images_segment1) < 2:
                images_segment1 = images
            if len(images_segment2) < 2:
                images_segment2 = images
                
            logger.info(f"Segmento 1: {len(images_segment1)} imágenes, Segmento 2: {len(images_segment2)} imágenes")
            
            # Aplicar optimización de coherencia entre segmentos
            if maintain_context:
                # Usar función de coherencia para prompts mejorados
                segment1_prompt = enhance_prompt_consistency(
                    base_prompt=base_prompt,
                    segment_number=1,
                    total_segments=2,
                    previous_context="",
                    dynamic_camera_changes=dynamic_camera_changes,
                    maintain_language=True
                )
                
                segment2_prompt = enhance_prompt_consistency(
                    base_prompt=base_prompt,
                    segment_number=2, 
                    total_segments=2,
                    previous_context="primer segmento con showcasing del producto y narración introductoria en el mismo idioma",
                    dynamic_camera_changes=dynamic_camera_changes,
                    maintain_language=True
                )
            else:
                # Prompts tradicionales simplificados 
                segment1_prompt = f"Professional product showcase - part 1: {base_prompt[:50]}"
                segment2_prompt = f"Professional product showcase - part 2: {base_prompt[:50]}"
            
            # Generar segmentos de video
            logger.info("Generando segmento 1/2...")
            segment1_result = await self.generate_video_from_images(
                images=images_segment1,
                prompt=segment1_prompt,
                duration_seconds=segment_duration,
                transition_style=transition_style,
                aspect_ratio=aspect_ratio,
                motion_strength=motion_strength,
                fps=fps,
                interpolation_frames=interpolation_frames,
                pan_direction=pan_direction,
                model_name=model_name,
                style="reel_optimized" if is_product_showcase else "social_media",
                is_product_showcase=is_product_showcase,
                maintain_context=maintain_context,
                add_narration=add_narration,
                text_overlays=text_overlays,
                dynamic_camera_changes=dynamic_camera_changes
            )
            
            logger.info("Generando segmento 2/2...")
            segment2_result = await self.generate_video_from_images(
                images=images_segment2,
                prompt=segment2_prompt,
                duration_seconds=segment_duration,
                transition_style=transition_style,
                aspect_ratio=aspect_ratio,
                motion_strength=motion_strength,
                fps=fps,
                interpolation_frames=interpolation_frames,
                pan_direction=pan_direction,
                model_name=model_name,
                style="reel_optimized" if is_product_showcase else "social_media",
                is_product_showcase=is_product_showcase,
                maintain_context=maintain_context,
                add_narration=add_narration,
                text_overlays=text_overlays,
                dynamic_camera_changes=dynamic_camera_changes
            )
            
            # Usar la función de procesamiento de video para concatenar
            return process_concatenated_video(
                segment_results=[segment1_result, segment2_result],
                segment_duration=segment_duration,
                fps=fps,
                aspect_ratio=aspect_ratio,
                transition_style=transition_style,
                motion_strength=motion_strength,
                total_images=len(images)
            )
                    
        except Exception as e:
            logger.error(f"Error en concatenación de video: {e}")
            raise Exception(f"Error generando video concatenado: {str(e)}")

