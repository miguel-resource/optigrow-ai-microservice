# OptiGrow AI Microservice

Microservicio en Python para consumir modelos de IA (actualmente Google Gemini) desde Laravel u otras aplicaciones.

## Características

- API REST con FastAPI
- Soporte para Google Gemini
- Arquitectura extensible para agregar más modelos
- Autenticación con API Key
- CORS configurado para Laravel
- Documentación automática con Swagger/OpenAPI
- Información de uso de tokens

## Requisitos

- Python 3.8+
- Cuenta de Google Cloud con acceso a Gemini API
- API Key de Gemini

## Instalación

### 1. Clonar el repositorio

```bash
cd optigrow-ai-microservice
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copia el archivo `.env.example` a `.env` y configura tus credenciales:

```bash
cp .env.example .env
```

Edita el archivo `.env`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TITLE=OptiGrow AI Microservice
API_VERSION=1.0.0

# Security
API_KEY=tu-clave-secreta-aqui

# Gemini Configuration
GEMINI_API_KEY=tu-api-key-de-gemini-aqui
GEMINI_MODEL=gemini-pro

# CORS Settings (Laravel URL)
ALLOWED_ORIGINS=http://localhost:8000,http://localhost:3000

# Environment
ENVIRONMENT=development
```

## Ejecución

### Modo desarrollo

```bash
python main.py
```

O usando uvicorn directamente:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Modo producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

La API estará disponible en `http://localhost:8000`

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

### 2. Crear un servicio en Laravel

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;

class AIService
{
    protected $baseUrl;
    protected $apiKey;

    public function __construct()
    {
        $this->baseUrl = config('services.ai_microservice.url');
        $this->apiKey = config('services.ai_microservice.key');
    }

    public function generate(string $prompt, string $model = 'gemini', array $options = [])
    {
        $response = Http::withHeaders([
            'X-API-Key' => $this->apiKey,
            'Content-Type' => 'application/json',
        ])->post("{$this->baseUrl}/api/v1/generate", [
            'prompt' => $prompt,
            'model' => $model,
            'temperature' => $options['temperature'] ?? 0.7,
            'max_tokens' => $options['max_tokens'] ?? null,
        ]);

        if ($response->successful()) {
            return $response->json();
        }

        throw new \Exception('Error al comunicarse con el microservicio de IA');
    }

    public function getAvailableModels()
    {
        $response = Http::withHeaders([
            'X-API-Key' => $this->apiKey,
        ])->get("{$this->baseUrl}/api/v1/models");

        return $response->json();
    }
}
```

### 3. Agregar configuración en `config/services.php`

```php
'ai_microservice' => [
    'url' => env('AI_MICROSERVICE_URL', 'http://localhost:8000'),
    'key' => env('AI_MICROSERVICE_KEY'),
],
```

### 4. Usar en un controlador

```php
<?php

namespace App\Http\Controllers;

use App\Services\AIService;
use Illuminate\Http\Request;

class AIController extends Controller
{
    protected $aiService;

    public function __construct(AIService $aiService)
    {
        $this->aiService = $aiService;
    }

    public function generate(Request $request)
    {
        $validated = $request->validate([
            'prompt' => 'required|string',
            'model' => 'string|in:gemini',
        ]);

        try {
            $result = $this->aiService->generate(
                $validated['prompt'],
                $validated['model'] ?? 'gemini'
            );

            return response()->json($result);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
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

### Generar texto

```bash
POST /api/v1/generate
Headers:
  X-API-Key: tu-clave-secreta
Content-Type: application/json

{
  "prompt": "¿Cuáles son los mejores consejos para cultivar tomates?",
  "model": "gemini",
  "temperature": 0.7,
  "max_tokens": 500
}
```

Respuesta:
```json
{
  "success": true,
  "model": "gemini",
  "text": "Aquí están los mejores consejos...",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### Listar modelos disponibles

```bash
GET /api/v1/models
Headers:
  X-API-Key: tu-clave-secreta
```

Respuesta:
```json
{
  "success": true,
  "models": ["gemini"]
}
```

## Testing

```bash
pytest tests/ -v
```

Con cobertura:

```bash
pytest tests/ --cov=app --cov-report=html
```

## Agregar nuevos modelos

Para agregar soporte para nuevos modelos de IA:

1. Crea una nueva clase en `app/services/` que herede de `BaseModelService`
2. Implementa los métodos abstractos requeridos
3. Registra el servicio en `ModelServiceFactory`
4. Agrega las variables de configuración necesarias en `.env` y `config/settings.py`

Ejemplo:

```python
# app/services/openai_service.py
from app.services.base_service import BaseModelService

class OpenAIService(BaseModelService):
    async def generate(self, prompt, **kwargs):
        # Implementación para OpenAI
        pass
    
    def validate_connection(self):
        # Validación de conexión
        pass
    
    @property
    def name(self):
        return "openai"
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

## Seguridad

- Todas las peticiones requieren autenticación mediante API Key
- Las API Keys se configuran en variables de entorno
- CORS está configurado para permitir solo orígenes específicos
- No expongas las claves API en el código fuente

## Licencia

Este proyecto está bajo la licencia MIT.

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Contacto

Para preguntas o soporte, contacta al equipo de desarrollo.
