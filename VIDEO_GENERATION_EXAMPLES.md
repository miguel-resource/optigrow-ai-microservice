# Generación de Videos con Veo 3.1

Este documento proporciona ejemplos de uso del endpoint de generación de videos usando el modelo Veo 3.1 de Google Gemini.

## Endpoint Principal

### `POST /generate-video`

Genera un video usando el modelo Veo de Gemini a partir de una descripción de texto y opcionalmente imágenes de referencia.

## Ejemplos de Uso

### 1. Generación Simple de Texto a Video

```json
{
  "prompt": "Un gato jugando en un jardín soleado, cinematográfico, 4K",
  "model": "veo-3.1-generate-preview",
  "aspect_ratio": "16:9",
  "resolution": "720p",
  "duration_seconds": 8
}
```

### 2. Generación con Prompt Negativo

```json
{
  "prompt": "Una mariposa volando por un bosque encantado al atardecer",
  "model": "veo-3.1-generate-preview",
  "aspect_ratio": "16:9",
  "resolution": "720p",
  "duration_seconds": 6,
  "negative_prompt": "cartoon, drawing, low quality, blurry, dark"
}
```

### 3. Generación con Imágenes de Referencia

```json
{
  "prompt": "Una mujer elegante caminando por una playa con el vestido ondeando al viento",
  "model": "veo-3.1-generate-preview",
  "aspect_ratio": "16:9",
  "resolution": "720p",
  "duration_seconds": 8,
  "reference_images": [
    "https://ejemplo.com/imagen1.jpg",
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",
    "https://ejemplo.com/imagen3.jpg"
  ]
}
```

### 4. Generación con Primer y Último Fotograma

```json
{
  "prompt": "Un automóvil deportivo acelerando por una carretera de montaña",
  "model": "veo-3.1-generate-preview",
  "first_frame": "https://ejemplo.com/primer_frame.jpg",
  "last_frame": "https://ejemplo.com/ultimo_frame.jpg",
  "aspect_ratio": "16:9",
  "resolution": "720p",
  "duration_seconds": 8
}
```

### 5. Video Vertical para Redes Sociales

```json
{
  "prompt": "Un chef preparando una receta paso a paso, estilo tutorial",
  "model": "veo-3.1-fast-preview",
  "aspect_ratio": "9:16",
  "resolution": "720p",
  "duration_seconds": 6
}
```

## Respuesta Típica

```json
{
  "success": true,
  "model": "veo-3.1-generate-preview",
  "video_uri": "gs://bucket/video.mp4",
  "operation_id": "projects/123/operations/456",
  "duration_seconds": 8,
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "usage": {
    "prompt_tokens": 15,
    "video_tokens": 1000,
    "total_tokens": 1015
  }
}
```

## Descarga de Videos

### `POST /download-video`

Una vez que el video está generado, puedes descargarlo usando el operation_id:

```json
{
  "operation_id": "projects/123/operations/456"
}
```

Este endpoint retornará el archivo de video directamente.

## Parámetros Disponibles

| Parámetro | Tipo | Descripción | Valores Permitidos |
|-----------|------|-------------|-------------------|
| `prompt` | string | Descripción del video (requerido) | Cualquier texto descriptivo |
| `model` | string | Modelo Veo a usar | `veo-3.1-generate-preview`, `veo-3.1-fast-preview` |
| `reference_images` | array | Imágenes de referencia (máx. 3) | URLs o strings base64 |
| `first_frame` | string | Imagen del primer fotograma | URL o string base64 |
| `last_frame` | string | Imagen del último fotograma | URL o string base64 |
| `aspect_ratio` | string | Relación de aspecto | `16:9`, `9:16` |
| `resolution` | string | Resolución del video | `720p`, `1080p` |
| `duration_seconds` | integer | Duración en segundos | 4, 6, 8 |
| `negative_prompt` | string | Elementos a evitar | Cualquier texto |

## Consejos para Mejores Resultados

### Escritura de Prompts Efectivos

1. **Sé específico y descriptivo**: Incluye detalles sobre sujeto, acción, estilo y ambiente
2. **Usa terminología cinematográfica**: "primer plano", "toma aérea", "cinematográfico"
3. **Especifica el estilo**: "realista", "animado", "cine negro", "futurista"
4. **Incluye detalles de iluminación**: "luz natural", "atardecer", "tonos cálidos"

### Ejemplos de Prompts Efectivos

- **Básico**: "Un perro corriendo en un parque"
- **Mejorado**: "Toma cinematográfica en cámara lenta de un golden retriever corriendo felizmente por un parque soleado con luz dorada del atardecer, enfoque en primer plano con fondo desenfocado"

### Uso de Imágenes de Referencia

- Máximo 3 imágenes de referencia por video
- Las imágenes deben ser de alta calidad
- Útil para mantener consistencia de personajes, objetos o estilos
- Funciona mejor con sujetos específicos (personas, productos, personajes)

### Optimización de Rendimiento

- Usa `veo-3.1-fast-preview` para generación más rápida
- `720p` es más rápido que `1080p`
- Videos más cortos (4-6s) se generan más rápido que 8s
- El aspect ratio `16:9` generalmente es más estable

## Limitaciones

- **Tiempo de generación**: Entre 11 segundos y 6 minutos
- **Retención**: Videos se eliminan del servidor después de 2 días
- **Marcas de agua**: Todos los videos incluyen marcas de agua SynthID
- **Restricciones regionales**: Algunas limitaciones en UE, Reino Unido, Suiza y MENA
- **Filtros de seguridad**: Contenido se filtra automáticamente por políticas de uso

## Códigos de Error Comunes

- `400`: Parámetros inválidos o video aún generándose
- `403`: API key inválida
- `404`: Operación no encontrada
- `500`: Error interno del servidor o problema con Gemini API

## Ejemplo Completo en Python

```python
import requests
import json

# Configuración
api_url = "https://tu-api.com"
api_key = "tu-api-key"

# Solicitud de generación
video_request = {
    "prompt": "Un atardecer mágico sobre el océano con gaviotas volando, cinematográfico",
    "model": "veo-3.1-generate-preview",
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration_seconds": 8,
    "negative_prompt": "cartoon, low quality"
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Generar video
response = requests.post(f"{api_url}/generate-video", 
                        json=video_request, 
                        headers=headers)

if response.status_code == 200:
    result = response.json()
    operation_id = result["operation_id"]
    print(f"Video generado! Operation ID: {operation_id}")
    
    # Descargar video
    download_request = {"operation_id": operation_id}
    download_response = requests.post(f"{api_url}/download-video", 
                                    json=download_request, 
                                    headers=headers)
    
    if download_response.status_code == 200:
        with open("video_generado.mp4", "wb") as f:
            f.write(download_response.content)
        print("Video descargado exitosamente!")
else:
    print(f"Error: {response.status_code} - {response.text}")
```