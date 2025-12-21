"""
OptiGrow AI Microservice
Microservicio para consumir diferentes modelos de IA desde Laravel
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from config.settings import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Microservicio para consumir diferentes modelos de IA desde Laravel",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(router, prefix="/api/v1")

# Evento de inicio
@app.on_event("startup")
async def startup_event():
    logger.info(f"Iniciando {settings.api_title} v{settings.api_version}")
    logger.info(f"Entorno: {settings.environment}")
    logger.info(f"Orígenes permitidos: {settings.origins_list}")


# Evento de cierre
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Cerrando aplicación")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz
    """
    return {
        "message": "OptiGrow AI Microservice",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    )
