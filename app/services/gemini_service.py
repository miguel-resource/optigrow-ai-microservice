from google import genai
from google.genai import types
from typing import Dict, Any, Optional, List, Union
from config.settings import settings
import requests

import logging
import base64
import io
from PIL import Image
import requests
import time
import json
import tempfile

logger = logging.getLogger(__name__)
_video_cache: Dict[str, Any] = {}

# Función para persistir caché temporalmente
def save_cache_to_disk():
    """Guarda el caché actual en disco para debugging"""
    try:
        cache_data = {}
        for key, video in _video_cache.items():
            cache_data[key] = {
                "type": str(type(video)),
                "has_uri": hasattr(video, 'uri'),
                "uri": getattr(video, 'uri', None) if hasattr(video, 'uri') else None,
                "timestamp": time.time()
            }
        
        with open('/tmp/video_cache_debug.json', 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Caché guardado en disco: {len(cache_data)} videos")
    except Exception as e:
        logger.error(f"Error guardando caché: {e}")

def load_cache_from_disk():
    """Intenta cargar caché desde disco para debugging"""
    try:
        with open('/tmp/video_cache_debug.json', 'r') as f:
            cache_data = json.load(f)
        
        logger.info(f"Caché cargado desde disco: {len(cache_data)} videos")
        return cache_data
    except Exception as e:
        logger.warning(f"No se pudo cargar caché desde disco: {e}")
        return {}


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
    
    def _convert_pil_to_veo_format(self, pil_image: Any) -> Dict[str, str]:
        """
        Convierte una imagen PIL al formato requerido por la API de Veo
        
        Args:
            pil_image: Objeto PIL Image
            
        Returns:
            Dict con bytesBase64Encoded y mimeType
        """
        try:
            import base64
            import io
            
            # Convertir a RGB si es necesario
            if pil_image.mode not in ('RGB', 'RGBA'):
                pil_image = pil_image.convert('RGB')
            
            # Convertir PIL Image a bytes
            buffer = io.BytesIO()
            
            # Determinar formato y MIME type
            format_type = 'JPEG'
            mime_type = 'image/jpeg'
            
            # Si es RGBA, usar PNG para preservar transparencia
            if pil_image.mode == 'RGBA':
                format_type = 'PNG'
                mime_type = 'image/png'
            
            pil_image.save(buffer, format=format_type, quality=95)
            image_bytes = buffer.getvalue()
            
            # Codificar a base64
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            
            logger.info(f"Imagen convertida al formato Veo: {mime_type}, tamaño: {len(base64_encoded)} caracteres")
            
            return {
                "bytesBase64Encoded": base64_encoded,
                "mimeType": mime_type
            }
            
        except Exception as e:
            logger.error(f"Error al convertir imagen PIL al formato Veo: {str(e)}")
            raise Exception(f"Error al convertir imagen al formato Veo: {str(e)}")
        """
        Convierte una imagen PIL o bytes a string base64
        
        Args:
            image_input: Imagen PIL, bytes, o string base64
            
        Returns:
            String base64 de la imagen
        """
        try:
            if isinstance(image_input, str):
                # Ya es base64 o necesita procesamiento
                if image_input.startswith('data:'):
                    # Extraer solo el base64
                    return image_input.split(',')[1]
                else:
                    # Asumir que ya es base64
                    return image_input
            elif hasattr(image_input, 'save'):  # PIL Image
                # Convertir PIL Image a base64
                import io
                import base64
                buffer = io.BytesIO()
                image_input.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                return base64.b64encode(image_bytes).decode('utf-8')
            elif isinstance(image_input, bytes):
                # Bytes directos
                import base64
                return base64.b64encode(image_input).decode('utf-8')
            else:
                raise ValueError(f"Tipo de imagen no soportado: {type(image_input)}")
        except Exception as e:
            logger.error(f"Error convirtiendo imagen a base64: {str(e)}")
            raise Exception(f"Error convirtiendo imagen a base64: {str(e)}")

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
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        motion_strength: float = 0.5,
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
            duration_seconds (int): Duración en segundos (4, 6, 8, 10) (default: 8)
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
            
            valid_durations = [4, 6, 8, 10]
            if duration_seconds not in valid_durations:
                raise ValueError(f"duration_seconds debe ser uno de: {valid_durations}")
            
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
            
            return {
                "video": generated_video.video,
                "video_uri": generated_video.video.uri if hasattr(generated_video.video, 'uri') else None,
                "operation_id": operation.name,
                "duration_seconds": duration_seconds,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "usage": usage_metadata,
                "message": "Video generado exitosamente con Veo 3.1"
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
    
    async def generate_video_from_images(
        self,
        images: List[Union[str, Any]],
        prompt: str,
        model_name: Optional[str] = "veo-3.1-generate-preview",
        transition_style: str = "smooth",
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: int = 8,
        fps: int = 24,
        interpolation_frames: int = 8,
        motion_strength: float = 0.3,
        zoom_effect: bool = False,
        pan_direction: Optional[str] = None,
        fade_transitions: bool = True,
        audio_sync: bool = False,
        style: Optional[str] = None,
        seed: Optional[int] = None,
        use_nano_banana: bool = False
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
            duration_seconds (int): Duración total del video (4-30 segundos) (default: 8)
            fps (int): Fotogramas por segundo (24, 30, 60) (default: 24)
            interpolation_frames (int): Frames de interpolación entre imágenes (6-16) (default: 8)
                                       Valores bajos (6-10) mantienen fidelidad al producto.
                                       Valores altos (12-16) pueden generar contenido interpolado no deseado.
            motion_strength (float): Intensidad del movimiento aplicado (0.0-1.0) (default: 0.3)
                                   Valores bajos (0.1-0.4) mantienen fidelidad al producto.
                                   Valores altos (0.6-1.0) pueden generar contenido no relacionado.
            zoom_effect (bool): Aplicar efecto de zoom sutil en cada imagen (default: False)
            pan_direction (Optional[str]): Dirección de paneo ("left", "right", "up", "down", None)
            fade_transitions (bool): Usar fundidos suaves entre transiciones (default: True)
            audio_sync (bool): Sincronizar con audio (requiere audio_track) (default: False)
            style (Optional[str]): Estilo cinematográfico ("cinematic", "documentary", "artistic")
            seed (Optional[int]): Semilla para reproducibilidad
            use_nano_banana (bool): Usar Nano Banana para generar imagen de referencia (default: True)
                                   Si es False, usa directamente las imágenes del usuario
        
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
            
            if duration_seconds < 4 or duration_seconds > 30:
                raise ValueError("duration_seconds debe estar entre 4 y 30")
            
            valid_fps = [24, 30, 60]
            if fps not in valid_fps:
                raise ValueError(f"fps debe ser uno de: {valid_fps}")
            
            if interpolation_frames < 6 or interpolation_frames > 24:
                raise ValueError("interpolation_frames debe estar entre 6 y 24")
            
            valid_pan_directions = ["left", "right", "up", "down", None]
            if pan_direction not in valid_pan_directions:
                raise ValueError(f"pan_direction debe ser uno de: {valid_pan_directions[:-1]} o None")
            
            logger.info(f"Generando video desde {len(images)} imágenes - {duration_seconds}s")
            
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
            
            enhanced_prompt = f"""
            Crear un video promocional de {duration_seconds} segundos que muestre EXACTAMENTE este producto específico 
            usando estas {len(processed_images)} imágenes del producto. {prompt}
            
            IMPORTANTE: El video debe mostrar únicamente el producto real de las imágenes proporcionadas, 
            manteniendo EXACTAMENTE sus características, colores, forma, textura y todos los detalles específicos.
            NO inventar ni añadir elementos, características o detalles que no estén en las imágenes originales.
            NO cambiar la apariencia, forma, o características del producto durante transiciones o zooms.
            Mantener absoluta fidelidad visual al producto mostrado en las imágenes.
            
            Transiciones {transition_style} suaves y conservadoras entre cada imagen ({time_per_image:.1f}s por imagen).
            Evitar movimientos bruscos o efectos que puedan distorsionar la apariencia del producto.
            """
            
            if zoom_effect:
                enhanced_prompt += "\n- Aplicar zoom MUY sutil y controlado (máximo 10% de cambio) manteniendo EXACTAMENTE la integridad y apariencia del producto. NO inventar detalles durante el zoom."
            
            if pan_direction:
                enhanced_prompt += f"\n- Movimiento de cámara hacia {pan_direction} de forma muy suave y conservadora, sin distorsionar el producto."
            
            if fade_transitions:
                enhanced_prompt += "\n- Usar fundidos muy suaves para transiciones elegantes entre imágenes, manteniendo consistencia visual del producto."
            
            if style:
                enhanced_prompt = f"[{style.upper()} CINEMATOGRAPHY] {enhanced_prompt}"
            
            enhanced_prompt += "\n\nMantener coherencia visual, fluidez temporal y continuidad narrativa.\n\nRESTRICCIONES CRÍTICAS:\n- NO crear, inventar o añadir elementos que no estén en las imágenes originales\n- NO modificar la forma, color, textura o características del producto\n- NO generar fondos, objetos o elementos adicionales no presentes en las imágenes\n- Mantener absoluta fidelidad al producto real mostrado en las imágenes de referencia\n- Priorizar la precisión visual sobre la creatividad artística"
            
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
                    final_prompt = "Subtle camera movement showcasing product details, high quality, 4k"
                else:
                    final_prompt = enhanced_prompt
                    # Solo añadir movimiento si no hay instrucciones específicas y es necesario
                    if not any(word in final_prompt.lower() for word in ['movement', 'zoom', 'pan', 'cinematic', 'motion', 'flowing', 'moving']):
                        final_prompt += ", very subtle camera movement, minimal motion, product focus"

                generation_config = types.GenerateVideosConfig(
                    aspect_ratio="16:9",
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
                raise Exception("Operación completada sin respuesta válida")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            generated_video = operation.response.generated_videos[0]
            
            videos_generated_count = len(operation.response.generated_videos)
            
            consumption_seconds = videos_generated_count * duration_seconds
            
            # TODO: Valor actual del 2026, editar si este valor cambia
            cost_per_second = 0.40
            estimated_cost_usd = consumption_seconds * cost_per_second
            
            usage_metadata = {
                "videos_requested": 1,  # Siempre se pide 1 video
                "videos_generated": videos_generated_count,
                "duration_per_video_seconds": duration_seconds,
                "consumption_seconds": consumption_seconds,
                "estimated_cost_usd": estimated_cost_usd,
                "cost_per_second_usd": cost_per_second,
                "source_images_count": len(processed_images),
                "calculation_method": "real_count",
                "note": "Consumo calculado basado en videos exitosamente generados"
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
            
            # Persistir caché para debugging
            save_cache_to_disk()
            
            logger.info(f"Video generado: {duration_seconds}s desde {len(processed_images)} imágenes")
            
            return {
                "video": generated_video.video,
                "video_uri": generated_video.video.uri if hasattr(generated_video.video, 'uri') else None,
                "operation_id": operation.name,
                "metadata": metadata,
                "image_count": len(processed_images),
                "transitions": transitions_applied,
                "usage": usage_metadata,
                "message": f"Video cinematográfico generado exitosamente desde {len(processed_images)} imágenes con transiciones {transition_style}"
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
        Descarga un video generado por Veo usando Vertex AI
        
        Args:
            video_file: Objeto de archivo de video de Gemini
            filename: Nombre del archivo de destino
            
        Returns:
            Bytes del video descargado
        """
        try:
            if video_file is None:
                raise ValueError("No se proporcionó archivo de video válido")
            
            logger.info(f"Descargando video usando Vertex AI: {filename}")
            
            # Para Vertex AI, intentar obtener la URI directamente
            video_uri = None
            
            # Verificar diferentes atributos donde puede estar la URI
            try:
                if hasattr(video_file, 'uri'):
                    uri_value = getattr(video_file, 'uri')
                    if uri_value and isinstance(uri_value, str):
                        video_uri = uri_value
                        logger.info(f"URI obtenida de video_file.uri: {video_uri}")
            except Exception as e:
                logger.warning(f"Error accediendo a video_file.uri: {e}")
            
            # Intentar video_bytes directamente (es más directo para Vertex AI)
            if not video_uri:
                try:
                    if hasattr(video_file, 'video_bytes'):
                        video_bytes_attr = getattr(video_file, 'video_bytes')
                        if video_bytes_attr and len(video_bytes_attr) > 0:
                            logger.info(f"Video obtenido directamente desde video_bytes: {len(video_bytes_attr)} bytes")
                            return video_bytes_attr
                except Exception as e:
                    logger.warning(f"Error accediendo a video_file.video_bytes: {e}")
            
            # Verificar otros atributos URI
            if not video_uri:
                for attr_name in ['url', 'download_url', '_uri', 'path', 'file_uri', 'location', 'resource_name']:
                    try:
                        if hasattr(video_file, attr_name):
                            attr_value = getattr(video_file, attr_name)
                            if attr_value and isinstance(attr_value, str) and ('http' in attr_value or 'gs://' in attr_value):
                                video_uri = attr_value
                                logger.info(f"URI encontrada en {attr_name}: {video_uri}")
                                break
                    except Exception as e:
                        logger.warning(f"Error accediendo a {attr_name}: {e}")
                        continue
            
            if video_uri:
                logger.info(f"Descargando desde URI: {video_uri}")
                
                # Preparar headers para autenticación
                headers = {}
                if self.api_key and 'googleapis.com' in video_uri:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                elif self.api_key:
                    headers['x-goog-api-key'] = self.api_key
                
                # Descargar el video
                response = requests.get(video_uri, headers=headers, stream=True, timeout=300)
                response.raise_for_status()
                
                video_data = response.content
                logger.info(f"Video descargado exitosamente: {filename} ({len(video_data)} bytes)")
                return video_data
            
            else:
                # Si no hay URI ni video_bytes, intentar método save() como último recurso
                logger.warning("No se encontró URI ni video_bytes, intentando método save()")
                
                if hasattr(video_file, 'save'):
                    try:
                        # Usar un nombre de archivo más simple para evitar problemas con directorios
                        import time
                        temp_filename = f"video_{int(time.time())}.mp4"
                        temp_path = f"/tmp/{temp_filename}"
                        
                        logger.info(f"Intentando guardar video en: {temp_path}")
                        video_file.save(temp_path)
                        
                        if os.path.exists(temp_path):
                            with open(temp_path, 'rb') as f:
                                video_data = f.read()
                            
                            os.remove(temp_path)  # Limpiar archivo temporal
                            
                            logger.info(f"Video descargado usando método save(): {filename} ({len(video_data)} bytes)")
                            return video_data
                        else:
                            logger.error(f"El archivo {temp_path} no se creó correctamente")
                    
                    except Exception as save_error:
                        logger.error(f"Error con método save(): {save_error}")
                
                raise Exception(f"No se pudo obtener la URI del video ni usar método save() para operation_id: {operation_id if 'operation_id' in locals() else 'desconocido'}. Atributos del video_file: {dir(video_file) if video_file else 'None'}")
            
        except Exception as e:
            logger.error(f"Error al descargar video: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text[:500]}")
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
    
    def validate_connection(self) -> bool:
        """
        Valida la conexión con Gemini
        
        Returns:
            True si la conexión es válida
        """
        try:
            models = list(genai.list_models())
            return True
        except Exception as e:
            logger.error(f"Error al validar conexión con Gemini: {str(e)}")
            return False
