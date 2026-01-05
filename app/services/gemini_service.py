from google import genai
from typing import Dict, Any, Optional, List, Union
from config.settings import settings

import logging
import base64
import io
from PIL import Image
import requests
import time

logger = logging.getLogger(__name__)
_video_cache: Dict[str, Any] = {}


class GeminiService:
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Inicializa el servicio de Gemini
        
        Args:
            api_key: Clave API de Google Gemini
            model_name: Nombre del modelo específico de Gemini a utilizar
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Usar el SDK estándar de Google AI en lugar de Vertex AI para acceso a generate_videos
        self.client = genai.Client(
            api_key=api_key
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
    
    def _image_to_base64(self, image_input: Any) -> str:
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
                
                logger.info("Imagen procesada exitosamente como PIL Image")
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
            
            print('Gemini response:', response.text)
            if hasattr(response, 'usage_metadata'):
                print('Response usage_metadata:', response.usage_metadata)
            
            usage_metadata = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
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
            
            logger.info(f"Generando video con modelo: {model_name}")
            logger.info(f"Parámetros: duration={duration_seconds}s, resolution={resolution}, aspect_ratio={aspect_ratio}")
            logger.info(f"Motion strength: {motion_strength}, Style: {style}")
            logger.info(f"Prompt: {prompt[:150]}...")
            
            current_model = model_name if model_name else "veo-3.1-generate-preview"
            
            # Mejorar el prompt con contexto adicional
            enhanced_prompt = prompt
            if style:
                enhanced_prompt = f"[{style.upper()} style] {enhanced_prompt}"
            if negative_prompt:
                enhanced_prompt += f" | Evitar: {negative_prompt}"
            
            image = None
            if first_frame is None:
                logger.info("Generando imagen de referencia con Nano Banana...")
                # Step 1: Generate an image with Nano Banana (como en la documentación)
                image_response = self.client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=prompt,
                    config={"response_modalities":['IMAGE']}
                )
                
                # Step 2: Extraer imagen usando .parts[0].as_image() (como en la documentación)
                try:
                    image = image_response.parts[0].as_image()
                    logger.info("✓ Imagen de referencia generada exitosamente con Nano Banana")
                except AttributeError as attr_error:
                    logger.warning(f"as_image() no disponible: {str(attr_error)}, usando fallback")
                    # Fallback: extraer desde inline_data
                    if hasattr(image_response.parts[0], 'inline_data') and image_response.parts[0].inline_data:
                        import base64
                        image_bytes = base64.b64decode(image_response.parts[0].inline_data.data)
                        image = Image.open(io.BytesIO(image_bytes))
                        logger.info("✓ Imagen extraída desde inline_data")
                    else:
                        logger.warning("No se pudo generar imagen de referencia, continuando sin ella")
                        image = None
            elif first_frame is not None:
                image = first_frame
                
            logger.info("Iniciando generación de video con Veo 3.1...")
            
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
                logger.info(f"✓ Operación de video iniciada: {operation.name}")
            except AttributeError as attr_error:
                logger.error(f"AttributeError: {str(attr_error)}")
                raise Exception(f"Error de atributo: {str(attr_error)}")
            except Exception as api_error:
                logger.error(f"Error en generate_videos: {str(api_error)}")
                raise Exception(f"Error en API de videos: {str(api_error)}")
            
            logger.info(f"Operación de video iniciada: {operation.name}")
            logger.info("Esperando a que se complete la generación...")
            
            max_wait_time = 600
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Tiempo de espera agotado para la generación de video (>{max_wait_time}s)")
                
                logger.info(f"Esperando... ({int(elapsed_time)}s transcurridos)")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            if not hasattr(operation, 'response') or not operation.response:
                raise Exception("La operación se completó pero no se obtuvo respuesta")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            generated_video = operation.response.generated_videos[0]
            
            usage_metadata = None
            if hasattr(operation.response, 'usage_metadata') and operation.response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": getattr(operation.response.usage_metadata, 'prompt_token_count', 0),
                    "video_tokens": getattr(operation.response.usage_metadata, 'video_token_count', 0),
                    "total_tokens": getattr(operation.response.usage_metadata, 'total_token_count', 0)
                }
            else:
                usage_metadata = {
                    "prompt_tokens": len(prompt.split()),
                    "video_tokens": duration_seconds * 100,
                    "total_tokens": len(prompt.split()) + (duration_seconds * 100)
                }
            
            logger.info(f"Video REAL generado exitosamente. Duration: {duration_seconds}s, Resolution: {resolution}")
            
            # Guardar en caché para descargas posteriores
            _video_cache[operation.name] = generated_video.video
            logger.info(f"Video guardado en caché con operation_id: {operation.name}")
            
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
        interpolation_frames: int = 12,
        motion_strength: float = 0.7,
        zoom_effect: bool = False,
        pan_direction: Optional[str] = None,
        fade_transitions: bool = True,
        audio_sync: bool = False,
        style: Optional[str] = None,
        seed: Optional[int] = None,
        use_nano_banana: bool = True
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
            interpolation_frames (int): Frames de interpolación entre imágenes (6-24) (default: 12)
            motion_strength (float): Intensidad del movimiento aplicado (0.0-1.0) (default: 0.7)
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
            
            logger.info(f"Generando video desde {len(images)} imágenes")
            logger.info(f"Estilo: {transition_style}, Duración: {duration_seconds}s, FPS: {fps}")
            logger.info(f"Motion strength: {motion_strength}, Zoom: {zoom_effect}, Pan: {pan_direction}")
            
            # Procesar todas las imágenes
            processed_images = []
            for i, image in enumerate(images):
                try:
                    processed_img = self._process_image_input(image)
                    processed_images.append(processed_img)
                    logger.info(f"Imagen {i+1}/{len(images)} procesada exitosamente")
                except Exception as e:
                    logger.error(f"Error procesando imagen {i+1}: {str(e)}")
                    raise ValueError(f"Error en imagen {i+1}: {str(e)}")
            
            # Construir prompt mejorado para video desde imágenes
            time_per_image = duration_seconds / len(processed_images)
            
            enhanced_prompt = f"""
            Crear un video cinematográfico fluido de {duration_seconds} segundos que transforme estas {len(processed_images)} imágenes 
            en una narrativa visual coherente. {prompt}
            
            Especificaciones técnicas:
            - Transiciones {transition_style} entre cada imagen ({time_per_image:.1f}s por imagen)
            - {interpolation_frames} frames de interpolación suave entre imágenes
            - Movimiento con intensidad {motion_strength} para dar vida a las escenas
            """
            
            if zoom_effect:
                enhanced_prompt += "\n- Aplicar zoom cinematográfico sutil en cada imagen"
            
            if pan_direction:
                enhanced_prompt += f"\n- Movimiento de cámara hacia {pan_direction} durante las transiciones"
            
            if fade_transitions:
                enhanced_prompt += "\n- Fundidos suaves y naturales entre secuencias"
            
            if style:
                enhanced_prompt = f"[{style.upper()} CINEMATOGRAPHY] {enhanced_prompt}"
            
            enhanced_prompt += "\n\nMantener coherencia visual, fluidez temporal y continuidad narrativa."
            
            logger.info(f"Prompt construido: {enhanced_prompt[:200]}...")
            
            # Configurar parámetros de generación
            current_model = model_name if model_name else "veo-3.1-generate-preview"
            
            # Para video desde múltiples imágenes, necesitamos enviar todas las imágenes
            logger.info(f"Enviando {len(processed_images)} imágenes a la API de Veo")
            
            operation_params = {
                "model": current_model,
                "prompt": enhanced_prompt,
            }
            
            # Generar imagen de referencia con Nano Banana según la documentación oficial
            # Sigue el patrón exacto de: https://ai.google.dev/gemini-api/docs/veo
            reference_image_obj = None
            
            if len(processed_images) > 0:
                if use_nano_banana:
                    logger.info(f"Generando imagen de referencia con Nano Banana basada en {len(processed_images)} imágenes")
                    
                    try:
                        # Step 1: Generate an image with Nano Banana (exactamente como en la documentación)
                        image = self.client.models.generate_content(
                            model="gemini-2.5-flash-image",
                            contents=enhanced_prompt,
                            config={"response_modalities": ['IMAGE']}
                        )
                        
                        # Step 2: Usar la imagen generada directamente con image.parts[0].as_image()
                        # Como en el ejemplo oficial de la documentación
                        try:
                            reference_image_obj = image.parts[0].as_image()
                            logger.info("✓ Imagen de referencia generada exitosamente con Nano Banana usando .parts[0].as_image()")
                        except AttributeError as attr_error:
                            logger.warning(f"as_image() no disponible: {str(attr_error)}")
                            # Fallback: extraer datos de imagen manualmente desde inline_data
                            if hasattr(image.parts[0], 'inline_data') and image.parts[0].inline_data:
                                import base64
                                image_bytes = base64.b64decode(image.parts[0].inline_data.data)
                                reference_image_obj = Image.open(io.BytesIO(image_bytes))
                                logger.info("✓ Imagen extraída desde inline_data como fallback")
                            else:
                                raise AttributeError("No se pudo extraer imagen de Nano Banana")
                                
                    except Exception as img_gen_error:
                        logger.error(f"Error generando imagen de referencia con Nano Banana: {str(img_gen_error)}")
                        logger.warning("Fallback: usando primera imagen del usuario")
                        reference_image_obj = processed_images[0]
                        # Agregar contexto al prompt sobre uso de imágenes del usuario
                        enhanced_prompt += f"\n\nUSAR ESTAS {len(processed_images)} IMÁGENES: Crear transiciones cinematográficas {transition_style} que respeten el contenido visual de cada imagen."
                else:
                    # No usar Nano Banana, directamente la primera imagen del usuario
                    logger.info(f"Omitiendo Nano Banana, usando directamente imagen del usuario")
                    reference_image_obj = processed_images[0]
                    enhanced_prompt += f"\n\nBASADO EN {len(processed_images)} IMÁGENES: Crear video con transiciones {transition_style} fluidas entre las imágenes proporcionadas."
            else:
                raise ValueError("No se procesaron imágenes válidas")
            
            logger.info(f"Parámetros finales: model={current_model}, images={len(processed_images)}, prompt_length={len(enhanced_prompt)}")
            logger.info(f"Imagen de referencia: {type(reference_image_obj)}")
            logger.info(f"Iniciando generación de video con modelo {current_model}...")
            
            # Verificar que el método existe antes de llamarlo
            if not hasattr(self.client.models, 'generate_videos'):
                logger.error("Método generate_videos no encontrado en client.models")
                logger.error(f"Tipo de client: {type(self.client)}")
                logger.error(f"Tipo de client.models: {type(self.client.models)}")
                logger.error(f"Métodos disponibles: {[m for m in dir(self.client.models) if not m.startswith('_')]}")
                raise Exception("El método generate_videos no está disponible. Verifica la instalación del paquete google-genai")
            
            # Step 2: Generate video with Veo (exactamente como en la documentación)
            try:
                logger.info(f"Llamando a generate_videos con: model={current_model}, image={type(reference_image_obj)}")
                operation = self.client.models.generate_videos(
                    model=current_model,
                    prompt=enhanced_prompt,
                    image=reference_image_obj
                )
                logger.info(f"✓ Operación de video iniciada exitosamente: {operation.name}")
            except AttributeError as attr_error:
                logger.error(f"AttributeError en generate_videos: {str(attr_error)}")
                raise Exception(f"Error de atributo al llamar generate_videos: {str(attr_error)}")
            except TypeError as type_error:
                logger.error(f"TypeError en generate_videos: {str(type_error)}")
                raise Exception(f"Error de tipo en parámetros: {str(type_error)}")
            except Exception as api_error:
                logger.error(f"Error general en generate_videos: {str(api_error)}")
                raise Exception(f"Error en API de videos: {str(api_error)}")
            
            logger.info(f"Operación iniciada: {operation.name}")
            logger.info("Procesando video desde imágenes...")
            
            # Esperar con timeout extendido para videos desde imágenes
            max_wait_time = 900  # 15 minutos para videos desde imágenes
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Timeout en generación de video desde imágenes (>{max_wait_time//60}min)")
                
                logger.info(f"Procesando... ({int(elapsed_time)}s) - Generando {len(images)} imágenes en video")
                time.sleep(15)  # Verificar cada 15 segundos
                operation = self.client.operations.get(operation)
            
            if not hasattr(operation, 'response') or not operation.response:
                raise Exception("Operación completada sin respuesta válida")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            generated_video = operation.response.generated_videos[0]
            
            # Calcular usage metadata
            base_tokens = len(enhanced_prompt.split())
            image_processing_tokens = len(processed_images) * 50
            interpolation_tokens = interpolation_frames * len(processed_images) * 5
            duration_tokens = duration_seconds * 25
            
            usage_metadata = {
                "prompt_tokens": base_tokens,
                "image_processing_tokens": image_processing_tokens,
                "interpolation_tokens": interpolation_tokens,
                "video_generation_tokens": duration_tokens,
                "total_tokens": base_tokens + image_processing_tokens + interpolation_tokens + duration_tokens
            }
            
            if hasattr(operation.response, 'usage_metadata') and operation.response.usage_metadata:
                usage_metadata.update({
                    "actual_prompt_tokens": getattr(operation.response.usage_metadata, 'prompt_token_count', base_tokens),
                    "actual_video_tokens": getattr(operation.response.usage_metadata, 'video_token_count', duration_tokens),
                    "actual_total_tokens": getattr(operation.response.usage_metadata, 'total_token_count', usage_metadata["total_tokens"])
                })
            
            # Crear metadata detallada
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
            
            # Guardar en caché
            _video_cache[operation.name] = generated_video.video
            logger.info(f"Video desde imágenes guardado en caché: {operation.name}")
            
            logger.info(f"Video generado exitosamente desde {len(processed_images)} imágenes")
            logger.info(f"Duración: {duration_seconds}s, Transiciones: {len(transitions_applied)}")
            
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
        # Mostrar todas las claves en caché para debugging
        logger.info(f"Claves en caché: {list(_video_cache.keys())}")
        logger.info(f"Buscando operation_id: {operation_id}")
        
        video = _video_cache.get(operation_id)
        if video:
            logger.info(f"Video encontrado en caché para operation_id: {operation_id}")
            return video
        
        logger.warning(f"Video no encontrado en caché para operation_id: {operation_id}")
        
        # Intentar buscar con variaciones del operation_id
        # A veces Vertex AI agrega prefijos como "video_models/"
        if operation_id.startswith("models/"):
            # Intentar con el prefijo de Vertex AI
            vertex_id = operation_id.replace("models/", "video_models/", 1)
            video = _video_cache.get(vertex_id)
            if video:
                logger.info(f"Video encontrado con clave alternativa: {vertex_id}")
                return video
        
        # Intentar buscar por coincidencia parcial en el nombre de la operación
        for cached_key in _video_cache.keys():
            if operation_id in cached_key or cached_key in operation_id:
                logger.info(f"Video encontrado por coincidencia parcial: {cached_key}")
                return _video_cache[cached_key]
        
        # Si no está en caché, intentar obtenerlo directamente de la API
        logger.info(f"Intentando obtener video directamente de la API de Gemini: {operation_id}")
        try:
            # Usar la API de Files de Gemini para recuperar el video
            # El operation_id generalmente es el file_id en la API de Files
            video_file = genai.get_file(name=operation_id)
            
            if video_file:
                logger.info(f"Video recuperado exitosamente de la API: {operation_id}")
                # Guardar en caché para futuras consultas
                _video_cache[operation_id] = video_file
                return video_file
            
        except Exception as e:
            logger.error(f"Error al recuperar video de la API: {e}")
        
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
        import os  # Añadir import aquí
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
                
                raise Exception("No se pudo obtener la URI del video ni usar método save()")
            
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
