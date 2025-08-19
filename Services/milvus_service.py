import asyncio
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from transformers import AutoTokenizer, AutoModel
import torch
import os


def convert_numpy_types(obj):
    """numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_milvus_hit_entity(hit):
    """Milvus Hit Entity를 JSON 직렬화 가능한 형태로 변환"""
    try:
        # hit.entity를 안전하게 변환
        entity_dict = {}
        if hasattr(hit, 'entity') and hit.entity:
            print(f"📋 [ENTITY] hit.entity 타입: {type(hit.entity)}")
            
            # entity가 딕셔너리인 경우
            if isinstance(hit.entity, dict):
                print(f"📋 [ENTITY] hit.entity 키들: {list(hit.entity.keys())}")
                for key, value in hit.entity.items():
                    try:
                        entity_dict[key] = convert_numpy_types(value)
                    except Exception as key_error:
                        print(f"⚠️ [ENTITY] 키 '{key}' 변환 실패: {str(key_error)}")
                        entity_dict[key] = str(value)  # 문자열로 변환
            # entity가 다른 타입인 경우 (Hit 객체 등)
            else:
                print(f"⚠️ [ENTITY] hit.entity가 딕셔너리가 아님: {type(hit.entity)}")
                # Hit 객체의 속성들을 직접 확인
                if hasattr(hit.entity, '__dict__'):
                    for attr_name, attr_value in hit.entity.__dict__.items():
                        if not attr_name.startswith('_'):  # private 속성 제외
                            try:
                                entity_dict[attr_name] = convert_numpy_types(attr_value)
                            except Exception as attr_error:
                                print(f"⚠️ [ENTITY] 속성 '{attr_name}' 변환 실패: {str(attr_error)}")
                                entity_dict[attr_name] = str(attr_value)
                else:
                    entity_dict = {"raw_entity": str(hit.entity)}
        
        return {
            "id": hit.id,
            "distance": convert_numpy_types(hit.distance),
            "entity": entity_dict
        }
    except Exception as e:
        # 변환 실패 시 기본 정보만 반환
        print(f"❌ [ENTITY] Entity 변환 중 오류: {str(e)}")
        return {
            "id": getattr(hit, 'id', None),
            "distance": getattr(hit, 'distance', None),
            "entity": {},
            "error": f"Entity 변환 실패: {str(e)}"
        }


class MilvusService:
    def __init__(self):
        self.connection = None
        self.tokenizer = None
        self.model = None
        self._initialize_connection()
        self._initialize_transformer()
    
    def _initialize_connection(self):
        """Milvus 연결 초기화"""
        try:
            host = os.getenv('MILVUS_HOST', 'localhost')
            port = os.getenv('MILVUS_PORT', '19530')
            
            connections.connect(
                alias="default",
                host=host,
                port=port
            )
            # 연결 확인
            self.connection = connections
            print(f"Milvus 연결 성공: {host}:{port}")
        except Exception as e:
            print(f"Milvus 연결 실패: {e}")
            raise
    
    def _initialize_transformer(self):
        """Transformer 모델 초기화"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print(f"Transformer 모델 로드 성공: {model_name}")
        except Exception as e:
            print(f"Transformer 모델 로드 실패: {e}")
            raise
    
    async def text_to_vector(self, text: str, target_dimension: Optional[int] = None) -> List[float]:
        """텍스트를 벡터로 변환"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                vector = embeddings.squeeze().numpy().tolist()
            
            print(f"원본 벡터 차원: {len(vector)}")
            
            # target_dimension이 지정된 경우 벡터 차원 조정
            if target_dimension is not None:
                print(f"목표 차원: {target_dimension}")
                if len(vector) > target_dimension:
                    # 차원이 큰 경우: 앞쪽부터 잘라내기
                    vector = vector[:target_dimension]
                    print(f"벡터 차원 축소: {len(vector)}")
                elif len(vector) < target_dimension:
                    # 차원이 작은 경우: 0으로 패딩
                    vector.extend([0.0] * (target_dimension - len(vector)))
                    print(f"벡터 차원 확장: {len(vector)}")
                else:
                    print(f"벡터 차원 일치: {len(vector)}")
            
            return vector
        except Exception as e:
            print(f"텍스트 벡터화 실패: {e}")
            raise
    
    async def create_collection(self, collection_name: str, dimension: int, 
                               metric_type: str = "COSINE", index_type: str = "IVF_FLAT", 
                               index_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """컬렉션 생성 API - PyMilvus 2.6.0에서 원하는 인덱스 타입으로 직접 생성"""
        try:
            print(f"📋 [CREATE] 컬렉션 생성 시작 - 컬렉션: {collection_name}")
            
            # 1단계: 컬렉션 존재 확인
            print(f"📋 [CREATE] 1단계: 컬렉션 존재 확인")
            if utility.has_collection(collection_name):
                print(f"❌ [CREATE] 컬렉션 '{collection_name}'이 이미 존재함")
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 이미 존재합니다."}
            print(f"✅ [CREATE] 컬렉션 '{collection_name}' 존재하지 않음 (생성 가능)")
            
            # 2단계: 매개변수 검증 및 기본값 설정
            print(f"📋 [CREATE] 2단계: 매개변수 검증 및 기본값 설정")
            
            # metric_type 검증
            valid_metric_types = ["L2", "IP", "COSINE"]
            if metric_type.upper() not in valid_metric_types:
                print(f"❌ [CREATE] 지원하지 않는 metric_type: {metric_type}")
                return {"success": False, "message": f"지원하지 않는 metric_type입니다. 지원되는 타입: {', '.join(valid_metric_types)}"}
            
            # index_type 검증
            valid_index_types = ["IVF_FLAT", "HNSW", "IVF_SQ8", "FLAT"]
            if index_type.upper() not in valid_index_types:
                print(f"⚠️ [CREATE] 지원하지 않는 index_type: {index_type}, 기본값 IVF_FLAT 사용")
                index_type = "IVF_FLAT"
            
            # index_params 기본값 설정
            if index_params is None:
                if index_type.upper() == "IVF_FLAT":
                    index_params = {"nlist": 1024}
                elif index_type.upper() == "HNSW":
                    index_params = {"M": 16, "efConstruction": 500}
                elif index_type.upper() == "IVF_SQ8":
                    index_params = {"nlist": 1024}
                else:  # FLAT
                    index_params = {}
            
            # PyMilvus 2.6.0+ 스타일의 인덱스 파라미터 구성
            index_params_obj = {
                "metric_type": metric_type.upper(),
                "index_type": index_type.upper(),
                "params": index_params
            }
            
            print(f"📋 [CREATE] 최종 인덱스 파라미터: {index_params_obj}")
            
            # 3단계: 컬렉션 생성과 인덱스 생성 - PyMilvus 2.6.0의 create_collection 사용
            print(f"📋 [CREATE] 3단계: 컬렉션 생성 및 인덱스 생성")
            try:
                # 환경변수에서 Milvus 연결 정보 가져오기
                host = os.getenv('MILVUS_HOST', 'localhost')
                port = os.getenv('MILVUS_PORT', '19530')
                uri = f"http://{host}:{port}"
                
                print(f"📋 [CREATE] Milvus 연결 시도: {uri}")
                
                from pymilvus import AsyncMilvusClient
                
                # AsyncMilvusClient 인스턴스 생성
                async_client = AsyncMilvusClient(uri=uri, token="")
                print(f"✅ [CREATE] AsyncMilvusClient 생성 성공")
                
                await async_client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    primary_field_name="id",
                    id_type="int",
                    vector_field_name="vector",
                    metric_type=metric_type.upper(),
                    auto_id=False,  # 수동 ID 할당을 위해 False로 변경
                    index_params=index_params_obj  # 인덱스를 함께 생성
                )
                print(f"✅ [CREATE] 컬렉션 '{collection_name}' 생성 성공")
                
            except Exception as e:
                print(f"❌ [CREATE] 컬렉션 생성 실패: {str(e)}")
                print(f"❌ [CREATE] 에러 타입: {type(e).__name__}")
                import traceback
                print(f"❌ [CREATE] 상세 에러: {traceback.format_exc()}")
                return {"success": False, "message": f"컬렉션 생성 실패: {str(e)}"}
            
            # 4단계: 컬렉션 로드
            print(f"📋 [CREATE] 4단계: 컬렉션 로드")
            try:
                await async_client.load_collection(collection_name)
                print(f"✅ [CREATE] 컬렉션 '{collection_name}' 로드 성공")
            except Exception as e:
                print(f"❌ [CREATE] 컬렉션 로드 실패: {str(e)}")
                # 로드 실패해도 컬렉션은 생성됨
                print(f"⚠️ [CREATE] 컬렉션 로드 실패했지만 컬렉션은 생성됨")
            
            print(f"🎉 [CREATE] 컬렉션 '{collection_name}' 생성 완료!")
            return {"success": True, "message": f"컬렉션 '{collection_name}' 생성 완료 (metric_type: {metric_type.upper()}, index_type: {index_type.upper()})"}
            
        except Exception as e:
            print(f"❌ [CREATE] 컬렉션 생성 중 예외 발생: {str(e)}")
            import traceback
            print(f"❌ [CREATE] 상세 에러: {traceback.format_exc()}")
            return {"success": False, "message": f"컬렉션 생성 실패: {str(e)}"}
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 삭제"""
        try:
            if not utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            
            utility.drop_collection(collection_name)
            return {"success": True, "message": f"컬렉션 '{collection_name}' 삭제 완료"}
        except Exception as e:
            return {"success": False, "message": f"컬렉션 삭제 실패: {e}"}
    
    async def get_collections(self) -> Dict[str, Any]:
        """모든 컬렉션 조회 - 상세 정보 포함"""
        try:
            print(f"📋 [COLLECTIONS] 컬렉션 목록 조회 시작")
            
            collection_names = utility.list_collections()
            if not collection_names:
                print(f"📭 [COLLECTIONS] 생성된 컬렉션이 없음")
                return {"success": True, "collections": []}
            
            print(f"📋 [COLLECTIONS] 발견된 컬렉션: {collection_names}")
            
            # 각 컬렉션의 상세 정보 수집
            collections_info = []
            for collection_name in collection_names:
                try:
                    print(f"📋 [COLLECTIONS] 컬렉션 '{collection_name}' 정보 수집 중...")
                    
                    # 컬렉션 객체 생성
                    collection = Collection(collection_name)
                    
                    # 기본 정보
                    collection_info = {
                        "name": collection_name,
                        "entity_count": collection.num_entities,
                        "index_type": "UNKNOWN",
                        "metric_type": "UNKNOWN",
                        "dimension": None
                    }
                    
                    # 인덱스 정보 조회
                    try:
                        index_info = collection.index()
                        if index_info:
                            if hasattr(index_info, 'metric_type'):
                                collection_info["metric_type"] = index_info.metric_type
                            elif hasattr(index_info, 'params') and isinstance(index_info.params, dict):
                                collection_info["metric_type"] = index_info.params.get('metric_type', 'UNKNOWN')
                                collection_info["index_type"] = index_info.params.get('index_type', 'UNKNOWN')
                    except Exception as e:
                        print(f"⚠️ [COLLECTIONS] 컬렉션 '{collection_name}' 인덱스 정보 조회 실패: {str(e)}")
                    
                    # 벡터 차원 정보 조회
                    try:
                        for field in collection.schema.fields:
                            if field.name == "vector" and hasattr(field, 'params') and field.params:
                                collection_info["dimension"] = field.params.get("dim")
                                break
                    except Exception as e:
                        print(f"⚠️ [COLLECTIONS] 컬렉션 '{collection_name}' 차원 정보 조회 실패: {str(e)}")
                    
                    collections_info.append(collection_info)
                    print(f"✅ [COLLECTIONS] 컬렉션 '{collection_name}' 정보 수집 완료")
                    
                except Exception as e:
                    print(f"❌ [COLLECTIONS] 컬렉션 '{collection_name}' 정보 수집 실패: {str(e)}")
                    # 기본 정보라도 포함
                    collections_info.append({
                        "name": collection_name,
                        "entity_count": 0,
                        "index_type": "UNKNOWN",
                        "metric_type": "UNKNOWN",
                        "dimension": None
                    })
            
            print(f"🎉 [COLLECTIONS] 컬렉션 목록 조회 완료: {len(collections_info)}개")
            return {"success": True, "collections": collections_info}
            
        except Exception as e:
            print(f"❌ [COLLECTIONS] 컬렉션 목록 조회 실패: {str(e)}")
            return {"success": False, "message": f"컬렉션 조회 실패: {str(e)}"}
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회 (metric_type 포함)"""
        try:
            print(f"📋 [INFO] 컬렉션 정보 조회 시작 - 컬렉션: {collection_name}")
            
            # 1단계: 컬렉션 존재 확인
            print(f"📋 [INFO] 1단계: 컬렉션 존재 확인")
            if not utility.has_collection(collection_name):
                print(f"❌ [INFO] 컬렉션 '{collection_name}'이 존재하지 않음")
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            print(f"✅ [INFO] 컬렉션 '{collection_name}' 존재 확인됨")
            
            # 2단계: 컬렉션 객체 생성
            print(f"📋 [INFO] 2단계: 컬렉션 객체 생성")
            try:
                collection = Collection(collection_name)
                print(f"✅ [INFO] 컬렉션 객체 생성 성공")
            except Exception as e:
                print(f"❌ [INFO] 컬렉션 객체 생성 실패: {str(e)}")
                return {"success": False, "message": f"컬렉션 객체 생성 실패: {str(e)}"}
            
            # 3단계: 인덱스 정보 조회
            print(f"📋 [INFO] 3단계: 인덱스 정보 조회")
            try:
                index_info = collection.index()
                metric_type = None
                if index_info:
                    # Index 객체의 속성들을 확인
                    print(f"📋 [INFO] 인덱스 객체 타입: {type(index_info)}")
                    print(f"📋 [INFO] 인덱스 객체 속성들: {dir(index_info)}")
                    
                    # metric_type을 안전하게 추출
                    if hasattr(index_info, 'metric_type'):
                        metric_type = index_info.metric_type
                    elif hasattr(index_info, 'params'):
                        print(f"📋 [INFO] index_info.params 타입: {type(index_info.params)}")
                        print(f"📋 [INFO] index_info.params 내용: {index_info.params}")
                        if hasattr(index_info.params, 'metric_type'):
                            metric_type = index_info.params.metric_type
                        elif isinstance(index_info.params, dict) and 'metric_type' in index_info.params:
                            metric_type = index_info.params['metric_type']
                        else:
                            metric_type = "UNKNOWN"
                            print(f"⚠️ [INFO] params에서 metric_type을 찾을 수 없음")
                    else:
                        metric_type = "UNKNOWN"
                        print(f"⚠️ [INFO] metric_type을 찾을 수 없음")
                    
                    print(f"✅ [INFO] 인덱스 정보 조회 성공 - metric_type: {metric_type}")
                else:
                    print(f"⚠️ [INFO] 인덱스 정보가 없음")
            except Exception as e:
                print(f"❌ [INFO] 인덱스 정보 조회 실패: {str(e)}")
                return {"success": False, "message": f"인덱스 정보 조회 실패: {str(e)}"}
            
            # 4단계: 스키마 정보 조회
            print(f"📋 [INFO] 4단계: 스키마 정보 조회")
            try:
                schema_info = {
                    "fields": [],
                    "description": collection.schema.description
                }
                
                for field in collection.schema.fields:
                    field_info = {
                        "name": field.name,
                        "dtype": str(field.dtype),
                        "is_primary": field.is_primary,
                        "auto_id": field.auto_id
                    }
                    if hasattr(field, 'params') and field.params:
                        field_info["params"] = field.params
                    schema_info["fields"].append(field_info)
                
                print(f"✅ [INFO] 스키마 정보 조회 성공 - 필드 개수: {len(schema_info['fields'])}")
            except Exception as e:
                print(f"❌ [INFO] 스키마 정보 조회 실패: {str(e)}")
                return {"success": False, "message": f"스키마 정보 조회 실패: {str(e)}"}
            
            print(f"🎉 [INFO] 컬렉션 정보 조회 완료 성공!")
            
            # dimension 정보 추출
            dimension = None
            for field in schema_info["fields"]:
                if field.get("name") == "vector" and "params" in field:
                    dimension = field["params"].get("dim")
                    break
            
            return {
                "success": True,
                "collection_name": collection_name,
                "metric_type": metric_type,
                "dimension": dimension,
                "collection_info": {
                    "collection_name": collection_name,
                    "metric_type": metric_type,
                    "dimension": dimension,
                    "schema": schema_info
                },
                "schema": schema_info
            }
        except Exception as e:
            import traceback
            error_message = str(e)
            error_traceback = traceback.format_exc()
            print(f"💥 [INFO] 예상치 못한 오류 발생: {error_message}")
            print(f"💥 [INFO] 상세 에러: {error_traceback}")
            return {
                "success": False, 
                "message": f"컬렉션 정보 조회 실패: {error_message}",
                "error_type": type(e).__name__,
                "error_traceback": error_traceback
            }
    
    async def insert_vectors(self, collection_name: str, data: List[Dict]) -> Dict[str, Any]:
        """벡터 데이터 삽입 - 시스템 필드만 사용 (id, vector)"""
        try:
            print(f"📋 [INSERT] 벡터 삽입 시작 - 컬렉션: {collection_name}")
            
            # 1단계: 컬렉션 존재 확인
            print(f"📋 [INSERT] 1단계: 컬렉션 존재 확인")
            if not utility.has_collection(collection_name):
                print(f"❌ [INSERT] 컬렉션 '{collection_name}'이 존재하지 않음")
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            print(f"✅ [INSERT] 컬렉션 '{collection_name}' 존재 확인됨")
            
            # 2단계: 컬렉션 로드
            print(f"📋 [INSERT] 2단계: 컬렉션 로드")
            collection = Collection(collection_name)
            collection.load()
            print(f"✅ [INSERT] 컬렉션 로드 성공")
            
            # 3단계: 벡터 차원 확인
            print(f"📋 [INSERT] 3단계: 벡터 차원 확인")
            vector_dimension = None
            for field in collection.schema.fields:
                if field.name == "vector":
                    vector_dimension = field.params.get("dim")
                    break
            
            if vector_dimension is None:
                print(f"❌ [INSERT] vector 필드의 차원을 찾을 수 없음")
                return {"success": False, "message": "vector 필드의 차원을 찾을 수 없습니다."}
            
            print(f"✅ [INSERT] 컬렉션 '{collection_name}'의 vector 차원: {vector_dimension}")
            
            # 4단계: 데이터 전처리 (시스템 필드만 사용)
            print(f"📋 [INSERT] 4단계: 데이터 전처리 (시스템 필드만 사용)")
            
            processed_data = []
            
            for i, item in enumerate(data):
                print(f"📋 [INSERT] 데이터 {i+1} 처리 중...")
                
                # text가 있으면 벡터로 변환
                if "text" in item:
                    try:
                        vector = await self.text_to_vector(item["text"], target_dimension=vector_dimension)
                        print(f"📋 [INSERT] 데이터 {i+1} 텍스트 -> 벡터 변환 성공 (차원: {len(vector)})")
                        
                        # 시스템 필드만 포함 (vector만)
                        cleaned_item = {"vector": vector}
                        
                        # 추가 메타데이터 필드들 로깅 (무시됨)
                        extra_fields = [key for key in item.keys() if key not in ["text", "vector"]]
                        if extra_fields:
                            print(f"⚠️ [INSERT] 데이터 {i+1} 추가 메타데이터 필드 무시: {extra_fields}")
                        
                        print(f"📋 [INSERT] 데이터 {i+1} 최종 필드: {list(cleaned_item.keys())}")
                        processed_data.append(cleaned_item)
                        
                    except Exception as e:
                        print(f"❌ [INSERT] 데이터 {i+1} 텍스트 -> 벡터 변환 실패: {str(e)}")
                        continue
                
                # vector가 직접 제공된 경우
                elif "vector" in item:
                    vector = item["vector"]
                    if len(vector) != vector_dimension:
                        print(f"❌ [INSERT] 데이터 {i+1} 벡터 차원 불일치: {len(vector)} != {vector_dimension}")
                        continue
                    
                    cleaned_item = {"vector": vector}
                    print(f"📋 [INSERT] 데이터 {i+1} 직접 벡터 사용 (차원: {len(vector)})")
                    processed_data.append(cleaned_item)
                
                else:
                    print(f"❌ [INSERT] 데이터 {i+1}에 text 또는 vector 필드가 없음")
                    continue
            
            if not processed_data:
                print(f"❌ [INSERT] 처리 가능한 데이터가 없음")
                return {"success": False, "message": "처리 가능한 데이터가 없습니다."}
            
            # 5단계: PyMilvus 형식으로 데이터 변환
            print(f"📋 [INSERT] 5단계: PyMilvus 형식으로 데이터 변환")
            
            # 현재 컬렉션의 벡터 개수를 확인하여 다음 ID 계산
            current_count = collection.num_entities
            print(f"📋 [INSERT] 현재 컬렉션 벡터 개수: {current_count}")
            
            # ID와 vector 필드를 별도로 처리
            id_values = []
            vector_values = []
            
            for i, item in enumerate(processed_data):
                # 0부터 시작하는 순차 ID 할당
                new_id = current_count + i
                id_values.append(new_id)
                vector_values.append(item["vector"])
                print(f"📋 [INSERT] 데이터 {i+1}에 ID {new_id} 할당")
            
            # PyMilvus 형식으로 데이터 구성 [id, vector]
            insert_data = [id_values, vector_values]
            
            print(f"📋 [INSERT] 삽입할 벡터 개수: {len(vector_values)}")
            print(f"📋 [INSERT] 할당된 ID 범위: {id_values[0]} ~ {id_values[-1]}")
            
            # 6단계: 데이터 삽입
            print(f"📋 [INSERT] 6단계: 데이터 삽입")
            collection.insert(insert_data)
            collection.flush()
            
            print(f"✅ [INSERT] 벡터 삽입 완료: {len(processed_data)}개")
            return {"success": True, "message": f"{len(processed_data)}개의 벡터 삽입 완료"}
            
        except Exception as e:
            print(f"❌ [INSERT] 벡터 삽입 중 예외 발생: {str(e)}")
            return {"success": False, "message": f"벡터 삽입 실패: {str(e)}"}
    
    async def reset_collection_ids(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션의 ID를 0부터 시작하도록 리셋 (주의: 모든 데이터 재삽입 필요)"""
        try:
            print(f"🔄 [RESET] 컬렉션 ID 리셋 시작 - 컬렉션: {collection_name}")
            
            # 1단계: 컬렉션 존재 확인
            if not utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            
            # 2단계: 기존 데이터 백업
            collection = Collection(collection_name)
            collection.load()
            
            # 모든 벡터 데이터 조회
            results = collection.query(expr="", output_fields=["*"], limit=10000)
            if not results:
                return {"success": False, "message": "리셋할 데이터가 없습니다."}
            
            print(f"📋 [RESET] 백업할 데이터 개수: {len(results)}")
            
            # 3단계: 기존 컬렉션 삭제
            utility.drop_collection(collection_name)
            print(f"✅ [RESET] 기존 컬렉션 삭제 완료")
            
            # 4단계: 새 컬렉션 생성 (auto_id=False)
            try:
                # 환경변수에서 Milvus 연결 정보 가져오기
                host = os.getenv('MILVUS_HOST', 'localhost')
                port = os.getenv('MILVUS_PORT', '19530')
                uri = f"http://{host}:{port}"
                
                print(f"📋 [RESET] Milvus 연결 시도: {uri}")
                
                from pymilvus import AsyncMilvusClient
                async_client = AsyncMilvusClient(uri=uri, token="")
                print(f"✅ [RESET] AsyncMilvusClient 생성 성공")
                
                # 컬렉션 스키마 정보 복원
                dimension = len(results[0]["vector"])
                
                await async_client.create_collection(
                    collection_name=collection_name,
                    dimension=dimension,
                    primary_field_name="id",
                    id_type="int",
                    vector_field_name="vector",
                    metric_type="COSINE",  # 기본값 사용
                    auto_id=False,  # 수동 ID 할당
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 1024}
                    }
                )
                print(f"✅ [RESET] 새 컬렉션 생성 완료")
                
            except Exception as e:
                print(f"❌ [RESET] 새 컬렉션 생성 실패: {str(e)}")
                print(f"❌ [RESET] 에러 타입: {type(e).__name__}")
                import traceback
                print(f"❌ [RESET] 상세 에러: {traceback.format_exc()}")
                return {"success": False, "message": f"새 컬렉션 생성 실패: {str(e)}"}
            
            # 5단계: 데이터 재삽입 (0부터 시작하는 ID)
            collection = Collection(collection_name)
            collection.load()
            
            id_values = list(range(len(results)))  # 0, 1, 2, ...
            vector_values = [item["vector"] for item in results]
            
            insert_data = [id_values, vector_values]
            collection.insert(insert_data)
            collection.flush()
            
            print(f"✅ [RESET] 데이터 재삽입 완료 - ID 범위: 0 ~ {len(results)-1}")
            return {"success": True, "message": f"컬렉션 ID 리셋 완료. 새로운 ID 범위: 0 ~ {len(results)-1}"}
            
        except Exception as e:
            print(f"❌ [RESET] ID 리셋 중 예외 발생: {str(e)}")
            return {"success": False, "message": f"ID 리셋 실패: {str(e)}"}
    
    async def delete_vectors(self, collection_name: str, ids: List[int]) -> Dict[str, Any]:
        """벡터 데이터 삭제"""
        try:
            if not utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            
            collection = Collection(collection_name)
            collection.load()
            
            expr = f"id in {ids}"
            collection.delete(expr)
            collection.flush()
            
            return {"success": True, "message": f"{len(ids)}개의 벡터 삭제 완료"}
        except Exception as e:
            return {"success": False, "message": f"벡터 삭제 실패: {e}"}
    
    async def get_vectors(self, collection_name: str, limit: int = 100) -> Dict[str, Any]:
        """벡터 데이터 조회"""
        try:
            if not utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            
            collection = Collection(collection_name)
            collection.load()
            
            results = collection.query(
                expr="",
                output_fields=["*"],
                limit=limit
            )
            
            # numpy 타입을 Python 기본 타입으로 변환
            converted_results = convert_numpy_types(results)
            
            return {"success": True, "vectors": converted_results}
        except Exception as e:
            return {"success": False, "message": f"벡터 조회 실패: {e}"}
    
    async def search_vectors(self, collection_name: str, query_text: str, 
                           search_params: Dict, limit: int = 10) -> Dict[str, Any]:
        """벡터 검색"""
        try:
            print(f"🔍 [SEARCH] 검색 시작 - 컬렉션: {collection_name}, 쿼리: '{query_text}', limit: {limit}")
            
            # 1단계: 컬렉션 존재 확인
            print(f"📋 [SEARCH] 1단계: 컬렉션 존재 확인")
            if not utility.has_collection(collection_name):
                print(f"❌ [SEARCH] 컬렉션 '{collection_name}'이 존재하지 않음")
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            print(f"✅ [SEARCH] 컬렉션 '{collection_name}' 존재 확인됨")
            
            # 2단계: 컬렉션 객체 생성
            print(f"📋 [SEARCH] 2단계: 컬렉션 객체 생성")
            try:
                collection = Collection(collection_name)
                print(f"✅ [SEARCH] 컬렉션 객체 생성 성공")
            except Exception as e:
                print(f"❌ [SEARCH] 컬렉션 객체 생성 실패: {str(e)}")
                return {"success": False, "message": f"컬렉션 객체 생성 실패: {str(e)}"}
            
            # 3단계: 컬렉션 로드
            print(f"📋 [SEARCH] 3단계: 컬렉션 로드")
            try:
                collection.load()
                print(f"✅ [SEARCH] 컬렉션 로드 성공")
            except Exception as e:
                print(f"❌ [SEARCH] 컬렉션 로드 실패: {str(e)}")
                return {"success": False, "message": f"컬렉션 로드 실패: {str(e)}"}
            
            # 4단계: 인덱스 정보 확인
            print(f"📋 [SEARCH] 4단계: 인덱스 정보 확인")
            try:
                index_info = collection.index()
                collection_metric_type = None
                if index_info:
                    # Index 객체의 속성들을 확인
                    print(f"📋 [SEARCH] 인덱스 객체 타입: {type(index_info)}")
                    print(f"📋 [SEARCH] 인덱스 객체 속성들: {dir(index_info)}")
                    
                    # metric_type을 안전하게 추출
                    if hasattr(index_info, 'metric_type'):
                        collection_metric_type = index_info.metric_type
                    elif hasattr(index_info, 'params'):
                        print(f"📋 [SEARCH] index_info.params 타입: {type(index_info.params)}")
                        print(f"📋 [SEARCH] index_info.params 내용: {index_info.params}")
                        if hasattr(index_info.params, 'metric_type'):
                            collection_metric_type = index_info.params.metric_type
                        elif isinstance(index_info.params, dict) and 'metric_type' in index_info.params:
                            collection_metric_type = index_info.params['metric_type']
                        else:
                            collection_metric_type = "UNKNOWN"
                            print(f"⚠️ [SEARCH] params에서 metric_type을 찾을 수 없음")
                    else:
                        collection_metric_type = "UNKNOWN"
                        print(f"⚠️ [SEARCH] metric_type을 찾을 수 없음")
                    
                    print(f"✅ [SEARCH] 인덱스 정보 확인됨 - metric_type: {collection_metric_type}")
                else:
                    print(f"⚠️ [SEARCH] 인덱스 정보가 없음")
            except Exception as e:
                print(f"❌ [SEARCH] 인덱스 정보 확인 실패: {str(e)}")
                return {"success": False, "message": f"인덱스 정보 확인 실패: {str(e)}"}
            
            # 5단계: metric_type 검증
            print(f"📋 [SEARCH] 5단계: metric_type 검증")
            requested_metric_type = search_params.get("metric_type", "L2")
            print(f"📋 [SEARCH] 요청된 metric_type: {requested_metric_type}, 컬렉션 metric_type: {collection_metric_type}")
            
            if collection_metric_type and requested_metric_type.upper() != collection_metric_type.upper():
                print(f"❌ [SEARCH] metric_type 불일치")
                return {
                    "success": False, 
                    "message": f"Metric type 불일치: 컬렉션 '{collection_name}'은 '{collection_metric_type}' 방식으로 생성되었습니다. "
                              f"검색 시에도 동일한 방식을 사용해야 합니다. "
                              f"현재 요청된 방식: '{requested_metric_type}', "
                              f"컬렉션 설정 방식: '{collection_metric_type}'",
                    "collection_metric_type": collection_metric_type,
                    "requested_metric_type": requested_metric_type,
                    "available_metric_types": ["L2", "IP", "COSINE"]
                }
            print(f"✅ [SEARCH] metric_type 검증 통과")
            
            # 6단계: 벡터 차원 확인
            print(f"📋 [SEARCH] 6단계: 벡터 차원 확인")
            try:
                vector_dimension = None
                for field in collection.schema.fields:
                    if field.name == "vector":
                        vector_dimension = field.params.get("dim")
                        break
                print(f"✅ [SEARCH] 벡터 차원 확인됨: {vector_dimension}")
            except Exception as e:
                print(f"❌ [SEARCH] 벡터 차원 확인 실패: {str(e)}")
                return {"success": False, "message": f"벡터 차원 확인 실패: {str(e)}"}
            
            # 7단계: 쿼리 벡터 생성
            print(f"📋 [SEARCH] 7단계: 쿼리 벡터 생성")
            try:
                query_vector = await self.text_to_vector(query_text, target_dimension=vector_dimension)
                print(f"✅ [SEARCH] 쿼리 벡터 생성 성공 - 차원: {len(query_vector)}")
            except Exception as e:
                print(f"❌ [SEARCH] 쿼리 벡터 생성 실패: {str(e)}")
                return {"success": False, "message": f"쿼리 벡터 생성 실패: {str(e)}"}
            
            # 8단계: 검색 파라미터 설정
            print(f"📋 [SEARCH] 8단계: 검색 파라미터 설정")
            try:
                params = search_params.get("params", {"nprobe": 10})
                search_params_final = {
                    "metric_type": requested_metric_type.upper(),
                    "params": params
                }
                print(f"✅ [SEARCH] 검색 파라미터 설정 완료: {search_params_final}")
            except Exception as e:
                print(f"❌ [SEARCH] 검색 파라미터 설정 실패: {str(e)}")
                return {"success": False, "message": f"검색 파라미터 설정 실패: {str(e)}"}
            
            # 9단계: Milvus 검색 실행
            print(f"📋 [SEARCH] 9단계: Milvus 검색 실행")
            try:
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params_final,
                    limit=limit,
                    output_fields=["*"]
                )
                print(f"✅ [SEARCH] Milvus 검색 실행 성공 - 결과 개수: {len(results) if results else 0}")
            except Exception as e:
                print(f"❌ [SEARCH] Milvus 검색 실행 실패: {str(e)}")
                import traceback
                print(f"❌ [SEARCH] 상세 에러: {traceback.format_exc()}")
                return {"success": False, "message": f"Milvus 검색 실행 실패: {str(e)}"}
            
            # 10단계: 결과 포맷팅
            print(f"📋 [SEARCH] 10단계: 결과 포맷팅")
            try:
                formatted_results = []
                for hits in results:
                    for hit in hits:
                        formatted_results.append(convert_milvus_hit_entity(hit))
                print(f"✅ [SEARCH] 결과 포맷팅 완료 - 포맷된 결과 개수: {len(formatted_results)}")
            except Exception as e:
                print(f"❌ [SEARCH] 결과 포맷팅 실패: {str(e)}")
                return {"success": False, "message": f"결과 포맷팅 실패: {str(e)}"}
            
            print(f"🎉 [SEARCH] 검색 완료 성공!")
            return {"success": True, "results": formatted_results}
            
        except Exception as e:
            import traceback
            error_message = str(e)
            error_traceback = traceback.format_exc()
            print(f"💥 [SEARCH] 예상치 못한 오류 발생: {error_message}")
            print(f"💥 [SEARCH] 상세 에러: {error_traceback}")
            
            if "metric type not match" in error_message.lower():
                return {
                    "success": False,
                    "message": f"Metric type 불일치 오류가 발생했습니다. 컬렉션 생성 시 설정한 거리 계산 방식과 "
                              f"검색 시 사용하는 방식이 다릅니다. 컬렉션 정보를 확인해주세요.",
                    "error_details": error_message,
                    "error_traceback": error_traceback
                }
            else:
                return {
                    "success": False, 
                    "message": f"벡터 검색 실패: {error_message}",
                    "error_type": type(e).__name__,
                    "error_traceback": error_traceback
                }


# 싱글톤 인스턴스
milvus_service = MilvusService() 