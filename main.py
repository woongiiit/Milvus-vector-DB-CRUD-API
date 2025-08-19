from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Routers import collection_router, vector_router

app = FastAPI(
    title="Milvus Vector DB API",
    description="Milvus Vector Database CRUD API with FastAPI",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(collection_router.router)
app.include_router(vector_router.router)

# 디버깅용 엔드포인트
@app.get("/debug/routes")
async def debug_routes():
    """등록된 라우터 정보 확인"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": [method for method in route.methods] if hasattr(route, 'methods') else [],
                "name": route.name if hasattr(route, 'name') else "unknown"
            })
    return {"routes": routes}


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Milvus Vector DB API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "message": "API 서버가 정상적으로 실행 중입니다."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 