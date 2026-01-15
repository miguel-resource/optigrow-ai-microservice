# OptiGrow AI Microservice

Microservicio en Python para consumir modelos de IA (Google Gemini) desde Laravel u otras aplicaciones. Incluye soporte para generación de texto y videos con IA.

## Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **Python 3.8+** - Lenguaje de programación
- **Google Generative AI** - SDK oficial de Gemini
- **Pydantic** - Validación y serialización de datos
- **Uvicorn** - Servidor ASGI de alto rendimiento
- **Pytest** - Framework de testing


## Documentación

Una vez iniciado el servidor, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Uso desde Laravel

### 1. Configurar en Laravel

En tu archivo `.env` de Laravel, agrega:

```env
AI_MICROSERVICE_URL=http://localhost:8000
AI_MICROSERVICE_KEY=tu-clave-secreta-aqui
```

### 2. Agregar configuración en `config/services.php`

```php
'ai_microservice' => [
    'url' => env('AI_MICROSERVICE_URL', 'http://localhost:8000'),
    'key' => env('AI_MICROSERVICE_KEY'),
],
```

## Endpoints de la API

### Health Check

```bash
GET /api/v1/health
```

Respuesta:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_available": ["gemini"]
}
```

### Generación de Texto

```bash
POST /api/v1/generate
Headers:
  X-API-Key: tu-clave-secreta
Content-Type: application/json

{
  "prompt": "¿Cuáles son los mejores consejos para cultivar tomates?",
  "model": "gemini",
  "temperature": 0.7,
  "max_tokens": 500,
  "top_p": 0.9,
  "top_k": 40
}
```

Respuesta:
```json
{
  "success": true,
  "model": "gemini",
  "text": "Aquí están los mejores consejos para cultivar tomates...",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 145,
    "total_tokens": 157
  }
}
```

### Generación de Videos

```bash
POST /api/v1/generate-video
Headers:
  X-API-Key: tu-clave-secreta
Content-Type: application/json

{
  "prompt": "Un gato jugando en un jardín soleado",
  "model": "veo-3.1-generate-preview",
  "duration_seconds": 8,
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "negative_prompt": "violencia, contenido inapropiado"
}
```

Respuesta:
```json
{
  "success": true,
  "model": "veo-3.1-generate-preview",
  "video": null,
  "video_uri": "https://example.com/simulated_video.mp4",
  "operation_id": "simulated_operation_1735512345",
  "duration_seconds": 8,
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "usage": {
    "prompt_tokens": 8,
    "video_tokens": 800,
    "total_tokens": 808
  },
  "message": "Video generado de forma simulada - APIs de video no disponibles en la librería estándar"
}
```

**Parámetros de Video:**
- `duration_seconds`: 15, 22, 29, 36, 43, 50, 57 segundos
- `resolution`: "720p" o "1080p" 
- `aspect_ratio`: "16:9" o "9:16"
- `reference_images`: Array de hasta 3 imágenes base64
- `first_frame`: Imagen para el primer fotograma
- `last_frame`: Imagen para el último fotograma

### Listar Modelos Disponibles

```bash
GET /api/v1/models
Headers:
  X-API-Key: tu-clave-secreta
```

Respuesta:
```json
{
  "success": true,
  "models": {
    "text": [
      "gemini-2.5-flash-lite",
      "gemini-2.5-flash", 
      "gemini-1.5-flash",
      "gemini-1.5-pro"
    ],
    "video": [
      "veo-3.1-generate-preview",
      "veo-3.1-fast-preview",
      "veo-3",
      "veo-3-fast"
    ]
  }
}
```

### Herramientas de Desarrollo

- **Linting**: `flake8 app/`
- **Formateo**: `black app/`
- **Tipo checking**: `mypy app/`
- **Documentación**: Swagger UI en `/docs`

### Variables de Entorno para Testing

```env
# .env.test
ENVIRONMENT=testing
GEMINI_API_KEY=test-key-for-mocking
API_KEY=test-api-key
```

## Estructura del proyecto

```
optigrow-ai-microservice/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py          # Middleware de autenticación
│   │   └── routes.py        # Endpoints de la API
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Modelos Pydantic
│   └── services/
│       ├── __init__.py
│       ├── base_service.py  # Clase base abstracta
│       ├── gemini_service.py # Implementación de Gemini
│       └── model_factory.py  # Factory pattern
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuración de la app
├── tests/
│   └── ...                  # Tests unitarios
├── .env.example             # Variables de entorno de ejemplo
├── .gitignore
├── main.py                  # Punto de entrada
├── requirements.txt         # Dependencias
└── README.md
```

```bash
# Activar logs detallados
export LOG_LEVEL=DEBUG
fastapi dev main.py
```
