from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/vector", tags=["vector"])

# Lazy loading for milvus_service
_milvus_service = None

def get_milvus_service():
    global _milvus_service
    if _milvus_service is None:
        try:
            from Services.milvus_service import milvus_service
            _milvus_service = milvus_service
        except Exception as e:
            print(f"Milvus service 초기화 실패: {e}")
            raise HTTPException(status_code=500, detail="Milvus 서비스 초기화 실패")
    return _milvus_service


class VectorData(BaseModel):
    text: Optional[str] = None
    vector: Optional[List[float]] = None
    # 추가 필드들은 동적으로 처리


class InsertVectorRequest(BaseModel):
    collection_name: str
    data: List[Dict[str, Any]]


class DeleteVectorRequest(BaseModel):
    collection_name: str
    ids: List[int]


class GetVectorsRequest(BaseModel):
    collection_name: str
    limit: Optional[int] = 100


class SearchParams(BaseModel):
    metric_type: str = "L2"  # L2, IP, COSINE
    params: Dict[str, Any] = {"nprobe": 10}


class VectorSearchRequest(BaseModel):
    collection_name: str
    query_text: str
    search_params: SearchParams
    limit: Optional[int] = 10


@router.post("/insert")
async def insert_vectors(request: InsertVectorRequest):
    """벡터 데이터 삽입 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.insert_vectors(
            collection_name=request.collection_name,
            data=request.data
        )
        
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 삽입 중 오류 발생: {str(e)}")


@router.post("/delete")
async def delete_vectors(request: DeleteVectorRequest):
    """벡터 데이터 삭제 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.delete_vectors(
            collection_name=request.collection_name,
            ids=request.ids
        )
        
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 삭제 중 오류 발생: {str(e)}")


@router.get("/vectors")
async def get_vectors(collection_name: str, limit: int = 100):
    """벡터 데이터 조회 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.get_vectors(
            collection_name=collection_name,
            limit=limit
        )
        
        if result["success"]:
            return {
                "status": "success",
                "vectors": result["vectors"],
                "count": len(result["vectors"])
            }
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 조회 중 오류 발생: {str(e)}")


@router.post("/search")
async def search_vectors(request: VectorSearchRequest):
    """벡터 검색 API"""
    try:
        print(f"🔍 [ROUTER] 벡터 검색 API 호출됨")
        print(f"📋 [ROUTER] 요청 데이터: collection_name={request.collection_name}, query_text='{request.query_text}', limit={request.limit}")
        
        milvus_service = get_milvus_service()
        search_params = request.search_params.dict()
        print(f"📋 [ROUTER] 검색 파라미터: {search_params}")
        
        print(f"📋 [ROUTER] MilvusService.search_vectors 호출 시작")
        result = await milvus_service.search_vectors(
            collection_name=request.collection_name,
            query_text=request.query_text,
            search_params=search_params,
            limit=request.limit
        )
        print(f"📋 [ROUTER] MilvusService.search_vectors 호출 완료")
        
        if result["success"]:
            print(f"✅ [ROUTER] 검색 성공 - 결과 개수: {len(result['results'])}")
            return {
                "status": "success",
                "results": result["results"],
                "count": len(result["results"])
            }
        else:
            print(f"❌ [ROUTER] 검색 실패 - 에러 메시지: {result['message']}")
            if "error_traceback" in result:
                print(f"❌ [ROUTER] 상세 에러: {result['error_traceback']}")
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"💥 [ROUTER] 예상치 못한 오류 발생: {str(e)}")
        print(f"💥 [ROUTER] 상세 에러: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"벡터 검색 중 오류 발생: {str(e)}") 