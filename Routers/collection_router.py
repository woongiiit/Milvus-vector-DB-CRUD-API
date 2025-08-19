from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/collection", tags=["collection"])

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


class CreateCollectionRequest(BaseModel):
    collection_name: str
    dimension: int
    metric_type: str = "L2"  # L2, IP, COSINE
    index_type: str = "IVF_FLAT"  # IVF_FLAT, IVF_SQ8, HNSW, FLAT
    index_params: Optional[Dict[str, Any]] = None  # 사용자 정의 인덱스 파라미터


class DeleteCollectionRequest(BaseModel):
    collection_name: str


@router.post("/create")
async def create_collection(request: CreateCollectionRequest):
    """컬렉션 생성 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.create_collection(
            collection_name=request.collection_name,
            dimension=request.dimension,
            metric_type=request.metric_type,
            index_type=request.index_type,
            index_params=request.index_params
        )
        
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"컬렉션 생성 중 오류 발생: {str(e)}")


@router.post("/delete")
async def delete_collection(request: DeleteCollectionRequest):
    """컬렉션 삭제 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.delete_collection(
            collection_name=request.collection_name
        )
        
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"컬렉션 삭제 중 오류 발생: {str(e)}")


@router.get("/collections")
async def get_collections():
    """모든 컬렉션 조회 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.get_collections()
        
        if result["success"]:
            return {
                "status": "success",
                "collections": result["collections"],
                "count": len(result["collections"])
            }
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        import traceback
        error_detail = f"컬렉션 조회 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # 서버 로그에 출력
        raise HTTPException(status_code=500, detail=f"컬렉션 조회 중 오류 발생: {str(e)}")


@router.get("/info/{collection_name}")
async def get_collection_info(collection_name: str):
    """컬렉션 정보 조회 API (metric_type 포함)"""
    try:
        print(f"📋 [ROUTER] 컬렉션 정보 조회 API 호출됨")
        print(f"📋 [ROUTER] 요청 데이터: collection_name={collection_name}")
        
        milvus_service = get_milvus_service()
        print(f"📋 [ROUTER] MilvusService.get_collection_info 호출 시작")
        result = await milvus_service.get_collection_info(collection_name)
        print(f"📋 [ROUTER] MilvusService.get_collection_info 호출 완료")
        
        if result["success"]:
            print(f"✅ [ROUTER] 컬렉션 정보 조회 성공")
            return {
                "status": "success",
                "collection_info": result
            }
        else:
            print(f"❌ [ROUTER] 컬렉션 정보 조회 실패 - 에러 메시지: {result['message']}")
            if "error_traceback" in result:
                print(f"❌ [ROUTER] 상세 에러: {result['error_traceback']}")
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"💥 [ROUTER] 예상치 못한 오류 발생: {str(e)}")
        print(f"💥 [ROUTER] 상세 에러: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"컬렉션 정보 조회 중 오류 발생: {str(e)}")


@router.post("/reset_ids")
async def reset_collection_ids(request: DeleteCollectionRequest):
    """컬렉션 ID를 0부터 시작하도록 리셋 API"""
    try:
        milvus_service = get_milvus_service()
        result = await milvus_service.reset_collection_ids(
            collection_name=request.collection_name
        )
        
        if result["success"]:
            return {"status": "success", "message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ID 리셋 중 오류 발생: {str(e)}") 