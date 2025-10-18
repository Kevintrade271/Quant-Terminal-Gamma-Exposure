from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import greeks, volatility

app = FastAPI(
    title="Quant Terminal API",
    description="API for Gamma Exposure and Volatility Analysis",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(greeks.router, prefix="/api", tags=["Greeks"])
app.include_router(volatility.router, prefix="/api", tags=["Volatility"])


@app.get("/")
async def root():
    return {
        "message": "Quant Terminal API v2.0",
        "docs": "/docs",
        "endpoints": {
            "greeks": "/api/greeks/{ticker}",
            "volatility": "/api/volatility/{ticker}",
            "status": "/api/status/{ticker}"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("""
╔════════════════════════════════════════════════════════════════╗
║             QUANT TERMINAL - API BACKEND                       ║
║                                                                ║
║  🚀 API iniciada en: http://localhost:8000                     ║
║  📚 Documentación: http://localhost:8000/docs                  ║
║                                                                ║
║  📊 Endpoints:                                                 ║
║     • GET /api/greeks/{ticker}                                 ║
║     • GET /api/volatility/{ticker}                             ║
║     • GET /api/status/{ticker}                                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
