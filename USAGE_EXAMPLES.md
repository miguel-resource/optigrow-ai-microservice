# Ejemplos de uso de la API OptiGrow AI

## 1. Generación de Texto con Gemini

### Usando cURL

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{
    "prompt": "¿Cuáles son los mejores consejos para cultivar tomates en invernadero?",
    "model": "gemini-2.5-flash",
    "temperature": 0.7
  }'
```

### Ejemplo con más parámetros

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{
    "prompt": "Explica el proceso de fotosíntesis en plantas",
    "model": "gemini-1.5-pro",
    "temperature": 0.5,
    "max_tokens": 500,
    "top_k": 40,
    "top_p": 0.95
  }'
```

## 2. Generación de Videos con Veo

### Generación básica de video

```bash
curl -X POST "http://localhost:8000/api/v1/generate-video" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{
    "prompt": "Un jardín de tomates creciendo bajo la luz del sol, cinematográfico, time-lapse",
    "model": "veo-3.1-generate-preview",
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration_seconds": 8
  }'
```

### Video con imágenes de referencia

```bash
curl -X POST "http://localhost:8000/api/v1/generate-video" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{
    "prompt": "Un granjero cuidando plantas en un invernadero moderno",
    "model": "veo-3.1-generate-preview",
    "reference_images": [
      "https://ejemplo.com/granjero.jpg",
      "https://ejemplo.com/invernadero.jpg"
    ],
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration_seconds": 6,
    "negative_prompt": "cartoon, drawing, low quality"
  }'
```

### Descarga de video generado

```bash
curl -X POST "http://localhost:8000/api/v1/download-video" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{
    "operation_id": "projects/123/operations/456"
  }' \
  --output video_generado.mp4
```

## 3. Usando Python (requests)

### Generación de Texto

```python
import requests
import json

# Configuración
API_URL = "http://localhost:8000/api/v1/generate"
API_KEY = "optigrow@admin-2025"

# Headers
headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Payload
data = {
    "prompt": "Dame 5 consejos para el cultivo de lechugas hidropónicas",
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 300
}

# Hacer la petición
response = requests.post(API_URL, headers=headers, json=data)

# Procesar respuesta
if response.status_code == 200:
    result = response.json()
    print("Respuesta del modelo:")
    print(result["text"])
    print("\nUso de tokens:", result.get("usage"))
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### Generación de Videos

```python
import requests
import time

# Configuración
API_URL = "http://localhost:8000/api/v1"
API_KEY = "optigrow@admin-2025"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Generar video
video_data = {
    "prompt": "Un campo de girasoles meciendo con el viento al atardecer, cinematográfico",
    "model": "veo-3.1-generate-preview",
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration_seconds": 8,
    "negative_prompt": "cartoon, drawing, low quality"
}

print("Generando video...")
response = requests.post(f"{API_URL}/generate-video", headers=headers, json=video_data)

if response.status_code == 200:
    result = response.json()
    operation_id = result["operation_id"]
    print(f"Video iniciado! Operation ID: {operation_id}")
    print(f"Duración: {result['duration_seconds']}s")
    print(f"Resolución: {result['resolution']}")
    print(f"Tokens utilizados: {result.get('usage', {})}")
    
    # Descargar video
    download_data = {"operation_id": operation_id}
    download_response = requests.post(f"{API_URL}/download-video", 
                                    headers=headers, 
                                    json=download_data)
    
    if download_response.status_code == 200:
        with open("video_girasoles.mp4", "wb") as f:
            f.write(download_response.content)
        print("Video descargado como 'video_girasoles.mp4'")
    else:
        print(f"Error al descargar: {download_response.status_code}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### Generación con Imágenes de Referencia

```python
import requests
import base64

# Función para convertir imagen a base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Configuración
API_URL = "http://localhost:8000/api/v1/generate-video"
API_KEY = "optigrow@admin-2025"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Preparar imágenes de referencia
reference_images = [
    f"data:image/jpeg;base64,{image_to_base64('granjero.jpg')}",
    "https://ejemplo.com/planta.jpg"  # También se pueden usar URLs
]

video_data = {
    "prompt": "Un granjero experto mostrando técnicas de cultivo en su invernadero",
    "model": "veo-3.1-generate-preview",
    "reference_images": reference_images,
    "aspect_ratio": "16:9",
    "resolution": "720p",
    "duration_seconds": 6
}

response = requests.post(API_URL, headers=headers, json=video_data)

if response.status_code == 200:
    result = response.json()
    print(f"Video con referencias generado! ID: {result['operation_id']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## 3. Usando JavaScript (Fetch API)

```javascript
const API_URL = "http://localhost:8000/api/v1/generate";
const API_KEY = "optigrow@admin-2025";

async function generateText(prompt) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
      },
      body: JSON.stringify({
        prompt: prompt,
        model: "gemini",
        temperature: 0.7,
        max_tokens: 500,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Respuesta:", data.text);
    console.log("Uso de tokens:", data.usage);
    return data;
  } catch (error) {
    console.error("Error:", error);
  }
}

// Usar la función
generateText("¿Cómo optimizar el riego en cultivos de precisión?");
```

## 4. Usando PHP (Laravel)

### Servicio AIService (app/Services/AIService.php)

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class AIService
{
    protected $baseUrl;
    protected $apiKey;

    public function __construct()
    {
        $this->baseUrl = config('services.ai_microservice.url');
        $this->apiKey = config('services.ai_microservice.key');
    }

    /**
     * Genera texto usando el microservicio de IA
     *
     * @param string $prompt
     * @param string $model
     * @param array $options
     * @return array
     * @throws \Exception
     */
    public function generate(string $prompt, string $model = 'gemini', array $options = [])
    {
        try {
            $response = Http::timeout(60)
                ->withHeaders([
                    'X-API-Key' => $this->apiKey,
                    'Content-Type' => 'application/json',
                ])
                ->post("{$this->baseUrl}/api/v1/generate", [
                    'prompt' => $prompt,
                    'model' => $model,
                    'temperature' => $options['temperature'] ?? 0.7,
                    'max_tokens' => $options['max_tokens'] ?? null,
                    'top_k' => $options['top_k'] ?? null,
                    'top_p' => $options['top_p'] ?? null,
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('Error en microservicio de IA', [
                'status' => $response->status(),
                'body' => $response->body()
            ]);

            throw new \Exception('Error al comunicarse con el microservicio de IA: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('Excepción en AIService', ['error' => $e->getMessage()]);
            throw $e;
        }
    }

    /**
     * Obtiene los modelos disponibles
     *
     * @return array
     */
    public function getAvailableModels()
    {
        $response = Http::withHeaders([
            'X-API-Key' => $this->apiKey,
        ])->get("{$this->baseUrl}/api/v1/models");

        return $response->json();
    }

    /**
     * Verifica el estado del servicio
     *
     * @return array
     */
    public function healthCheck()
    {
        $response = Http::get("{$this->baseUrl}/api/v1/health");
        return $response->json();
    }
}
```

### Controlador (app/Http/Controllers/AIController.php)

```php
<?php

namespace App\Http\Controllers;

use App\Services\AIService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class AIController extends Controller
{
    protected $aiService;

    public function __construct(AIService $aiService)
    {
        $this->aiService = $aiService;
    }

    /**
     * Genera texto usando IA
     */
    public function generate(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'prompt' => 'required|string|max:4000',
            'model' => 'nullable|string|in:gemini',
            'temperature' => 'nullable|numeric|min:0|max:2',
            'max_tokens' => 'nullable|integer|min:1|max:2048',
        ]);

        try {
            $result = $this->aiService->generate(
                $validated['prompt'],
                $validated['model'] ?? 'gemini',
                [
                    'temperature' => $validated['temperature'] ?? 0.7,
                    'max_tokens' => $validated['max_tokens'] ?? null,
                ]
            );

            return response()->json($result);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Ejemplo de uso específico: análisis de cultivos
     */
    public function analyzeCrop(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'crop_type' => 'required|string',
            'conditions' => 'required|array',
        ]);

        $prompt = "Analiza las siguientes condiciones para el cultivo de {$validated['crop_type']}: " .
                  json_encode($validated['conditions']) .
                  ". Proporciona recomendaciones específicas.";

        try {
            $result = $this->aiService->generate($prompt, 'gemini', [
                'temperature' => 0.5,
                'max_tokens' => 800,
            ]);

            return response()->json([
                'success' => true,
                'crop_type' => $validated['crop_type'],
                'analysis' => $result['text'],
                'usage' => $result['usage'],
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
```

### Rutas (routes/api.php)

```php
use App\Http\Controllers\AIController;

Route::prefix('ai')->group(function () {
    Route::post('/generate', [AIController::class, 'generate']);
    Route::post('/analyze-crop', [AIController::class, 'analyzeCrop']);
});
```

### Configuración (config/services.php)

```php
'ai_microservice' => [
    'url' => env('AI_MICROSERVICE_URL', 'http://localhost:8000'),
    'key' => env('AI_MICROSERVICE_KEY'),
],
```

### Variables de entorno Laravel (.env)

```env
AI_MICROSERVICE_URL=http://localhost:8000
AI_MICROSERVICE_KEY=optigrow@admin-2025
```

## 5. Usando Postman

1. **Método**: POST
2. **URL**: `http://localhost:8000/api/v1/generate`
3. **Headers**:
   - `Content-Type`: `application/json`
   - `X-API-Key`: `optigrow@admin-2025`
4. **Body** (raw JSON):
```json
{
  "prompt": "Describe las mejores prácticas para agricultura de precisión",
  "model": "gemini",
  "temperature": 0.7,
  "max_tokens": 500
}
```

## Modelos de Gemini disponibles

Para usar diferentes modelos de Gemini, actualiza `GEMINI_MODEL` en tu `.env`:

```env
# Opciones disponibles:
GEMINI_MODEL=gemini-pro                    # Modelo estándar
GEMINI_MODEL=gemini-2.0-flash-exp         # Gemini 2.0 Flash (experimental)
GEMINI_MODEL=gemini-1.5-flash             # Gemini 1.5 Flash (más rápido)
GEMINI_MODEL=gemini-1.5-pro               # Gemini 1.5 Pro (más potente)
```

## Parámetros disponibles

- **prompt** (requerido): El texto de entrada
- **model** (opcional): Nombre del modelo (default: "gemini")
- **temperature** (opcional): Controla la creatividad (0.0-2.0, default: 0.7)
  - 0.0: Más determinístico
  - 1.0: Balance
  - 2.0: Más creativo
- **max_tokens** (opcional): Máximo de tokens a generar
- **top_k** (opcional): Top-k sampling
- **top_p** (opcional): Top-p (nucleus) sampling (0.0-1.0)

## Respuesta de la API

```json
{
  "success": true,
  "model": "gemini",
  "text": "Aquí está la respuesta generada...",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}
```

## Ejemplo completo de prueba rápida

```bash
# 1. Verifica que el servidor esté corriendo
curl http://localhost:8000/api/v1/health

# 2. Haz una petición simple
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: optigrow@admin-2025" \
  -d '{"prompt": "Hola, ¿cómo estás?", "model": "gemini"}'
```
