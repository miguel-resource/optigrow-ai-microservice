"""
Servicio para interactuar con Google Gemini
"""
from google import genai
from google.genai.types import HttpOptions
from typing import Dict, Any, Optional, List, Union
import logging
import base64
import io
from PIL import Image
import requests
import time

logger = logging.getLogger(__name__)


class GeminiService:
    """Servicio para interactuar con Google Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Inicializa el servicio de Gemini
        
        Args:
            api_key: Clave API de Google Gemini
            model_name: Nombre del modelo específico de Gemini a utilizar
        """
        self.api_key = api_key
        self.model_name = model_name
        
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1")
        )
        logger.info(f"GeminiService inicializado con modelo: {model_name}")
    
    def _process_image_input(self, image_input: Union[str, Any]) -> Any:
        """
        Procesa una imagen de entrada que puede ser URL, base64 o objeto Image
        
        Args:
            image_input: URL, string base64, o objeto de imagen
            
        Returns:
            Objeto Image procesado para usar con Gemini
        """
        try:
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    # Es una URL
                    response = requests.get(image_input)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                elif image_input.startswith('data:image'):
                    # Es base64 con prefijo data URI
                    header, data = image_input.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # Asume que es base64 sin prefijo
                    image_data = base64.b64decode(image_input)
                    image = Image.open(io.BytesIO(image_data))
                
                return image
            else:
                # Asume que ya es un objeto Image o compatible
                return image_input
                
        except Exception as e:
            logger.error(f"Error al procesar imagen: {str(e)}")
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
            
            generation_config = {}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            if temperature is not None:
                generation_config["temperature"] = temperature
            if top_p is not None:
                generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
            
            response = self.client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=generation_config if generation_config else None
            )
            
            print('Gemini response:', response.text)
            print('Response usage_metadata:', response.usage_metadata)
            
            usage_metadata = None
            if response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count, #Inputs tokens
                    "completion_tokens": response.usage_metadata.candidates_token_count, #Outputs tokens 
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
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera un video usando el modelo Veo de Gemini
        
        Args:
            prompt: Descripción de texto del video a generar
            model_name: Modelo Veo a utilizar
            reference_images: Lista de imágenes de referencia (hasta 3)
            first_frame: Imagen para el primer fotograma
            last_frame: Imagen para el último fotograma 
            aspect_ratio: Relación de aspecto (16:9 o 9:16)
            resolution: Resolución del video (720p o 1080p)
            duration_seconds: Duración en segundos (4, 6 u 8)
            negative_prompt: Elementos que no se quieren en el video
        
        Returns:
            Diccionario con información del video generado y uso de tokens
        """
        try:
            # Primero verificar qué modelos están disponibles
            pager = self.client.models.list()
            available_models = [m.name for m in pager]
            print("🔍 Modelos disponibles:")
            for model in available_models:
                print(f"  - {model}")
            
            # Verificar si hay modelos Veo disponibles
            veo_models = [m for m in available_models if 'veo' in m.lower()]
            if veo_models:
                print(f"✅ Modelos Veo encontrados: {veo_models}")
            else:
                print("❌ No se encontraron modelos Veo disponibles")
            
            # Usar la API real de Veo para generar videos (versión simplificada)
            logger.info(f"Intentando generar video con modelo: {model_name}")
            logger.info(f"Parámetros recibidos: duration={duration_seconds}, resolution={resolution}, aspect_ratio={aspect_ratio}")
            
            import time
            from google.genai import types
            
            current_model = model_name if model_name else "veo-3.1-generate-preview"
            
            # Usar prompt simple por defecto para testing
            test_prompt = prompt if prompt else "A close up of two people staring at a cryptic drawing on a wall, torchlight flickering."
            
            logger.info(f"Iniciando operación de generación con Gemini API...")
            logger.info(f"Usando prompt: {test_prompt[:100]}...")
            
            # Operación simplificada como en el ejemplo
            operation = self.client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=test_prompt,
            )
            
            logger.info(f"Operación iniciada: {operation.name}")
            logger.info("Esperando a que se complete la generación del video...")
            
            # Sondear el estado de la operación hasta que esté completo
            max_wait_time = 600  # 10 minutos máximo
            start_time = time.time()
            
            while not operation.done:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    raise Exception(f"Tiempo de espera agotado para la generación de video (>{max_wait_time}s)")
                
                logger.info(f"Esperando... ({int(elapsed_time)}s transcurridos)")
                time.sleep(10)  # Esperar 10 segundos antes del siguiente sondeo
                operation = self.client.operations.get(operation)
            
            # Verificar si la operación fue exitosa
            if not hasattr(operation, 'response') or not operation.response:
                raise Exception("La operación se completó pero no se obtuvo respuesta")
                
            if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                raise Exception("No se generaron videos en la respuesta")
            
            # Obtener el video generado
            generated_video = operation.response.generated_videos[0]
            
            # Extraer información de uso si está disponible
            usage_metadata = None
            if hasattr(operation.response, 'usage_metadata') and operation.response.usage_metadata:
                usage_metadata = {
                    "prompt_tokens": getattr(operation.response.usage_metadata, 'prompt_token_count', 0),
                    "video_tokens": getattr(operation.response.usage_metadata, 'video_token_count', 0),
                    "total_tokens": getattr(operation.response.usage_metadata, 'total_token_count', 0)
                }
            else:
                # Estimación si no hay metadatos reales
                usage_metadata = {
                    "prompt_tokens": len(test_prompt.split()),
                    "video_tokens": duration_seconds * 100,
                    "total_tokens": len(test_prompt.split()) + (duration_seconds * 100)
                }
            
            logger.info(f"Video generado exitosamente. Duration: {duration_seconds}s, Resolution: {resolution}")
            
            return {
                "video": generated_video.video,
                "video_uri": generated_video.video.uri if hasattr(generated_video.video, 'uri') else None,
                "operation_id": operation.name,
                "duration_seconds": duration_seconds,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "usage": usage_metadata
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error al generar video con Veo: {error_message}")
            if hasattr(e, '__dict__'):
                logger.error(f"Detalles del error: {e.__dict__}")
            
            # Manejar diferentes tipos de errores de manera específica
            if "429" in error_message and "RESOURCE_EXHAUSTED" in error_message:
                raise Exception("❌ Límite de cuota excedido para Veo 3.1. El modelo está disponible pero se han agotado los créditos/límites de uso. Verifica tu plan en https://ai.dev/usage")
            elif "404" in error_message or "NOT_FOUND" in error_message:
                print("Modelo no encontrado el error message es:", error_message)
                raise Exception("❌ Modelo Veo no encontrado. Puede no estar disponible en tu región o cuenta.")
            elif "403" in error_message or "PERMISSION_DENIED" in error_message:
                raise Exception("❌ Acceso denegado al modelo Veo. Verifica que tu API key tenga permisos para usar modelos de video.")
            elif "401" in error_message or "UNAUTHENTICATED" in error_message:
                raise Exception("❌ API key inválida o expirada.")
            else:
                raise Exception(f"❌ Error de Veo API: {error_message}")
    
    async def download_video(self, video_file, filename: str = "generated_video.mp4") -> bytes:
        """
        Descarga un video generado por Veo
        
        Args:
            video_file: Objeto de archivo de video de Gemini
            filename: Nombre del archivo de destino
            
        Returns:
            Bytes del video descargado
        """
        try:
            # Descargar el video usando la API real de Gemini
            logger.info(f"Descargando video real desde Gemini: {filename}")
            
            # Descargar el video usando el cliente
            video_data = self.client.files.download(file=video_file)
            
            logger.info(f"Video descargado exitosamente: {filename} ({len(video_data)} bytes)")
            return video_data
            
        except Exception as e:
            logger.error(f"Error al descargar video: {str(e)}")
            raise Exception(f"Error al descargar video: {str(e)}")
        
    def validate_connection(self) -> bool:
        """
        Valida la conexión con Gemini
        
        Returns:
            True si la conexión es válida
        """
        try:
            # Usar el nuevo cliente para validar
            models = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Error al validar conexión con Gemini: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Retorna la lista de modelos de Gemini disponibles"""
        return [
            # Modelos de texto de Gemini
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            # Modelos de Video de Gemini (Veo)
            "veo-3.1-generate-preview",
            "veo-3.1-fast-preview",
            "veo-3",
            "veo-3-fast",
            "veo-2",
        ]
