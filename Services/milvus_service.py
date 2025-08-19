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
    
    async def create_collection(self, collection_name: str, dimension: int, schema_fields: List[Dict], metric_type: str = "L2") -> Dict[str, Any]:
        """컬렉션 생성"""
        try:
            if utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 이미 존재합니다."}
            
            # metric_type 검증
            valid_metric_types = ["L2", "IP", "COSINE"]
            if metric_type.upper() not in valid_metric_types:
                return {"success": False, "message": f"지원하지 않는 metric_type입니다. 지원되는 타입: {', '.join(valid_metric_types)}"}
            
            # 기본 필드들
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            # 사용자 정의 필드들 추가
            for field in schema_fields:
                field_name = field.get("name")
                field_type = field.get("type", "VARCHAR")
                field_max_length = field.get("max_length", 65535)
                
                if field_type.upper() == "VARCHAR":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=field_max_length))
                elif field_type.upper() == "INT64":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
                elif field_type.upper() == "FLOAT":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT))
                elif field_type.upper() == "DOUBLE":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.DOUBLE))
                elif field_type.upper() == "BOOL":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.BOOL))
            
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # 인덱스 생성
            index_params = {
                "metric_type": metric_type.upper(),
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
            return {"success": True, "message": f"컬렉션 '{collection_name}' 생성 완료 (metric_type: {metric_type.upper()})"}
        except Exception as e:
            return {"success": False, "message": f"컬렉션 생성 실패: {e}"}
    
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
        """모든 컬렉션 조회"""
        try:
            collections = utility.list_collections()
            return {"success": True, "collections": collections}
        except Exception as e:
            return {"success": False, "message": f"컬렉션 조회 실패: {e}"}
    
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
            return {
                "success": True,
                "collection_name": collection_name,
                "metric_type": metric_type,
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
        """벡터 데이터 삽입"""
        try:
            if not utility.has_collection(collection_name):
                return {"success": False, "message": f"컬렉션 '{collection_name}'이 존재하지 않습니다."}
            
            collection = Collection(collection_name)
            collection.load()
            
            # collection의 vector 필드 차원 확인
            vector_dimension = None
            for field in collection.schema.fields:
                if field.name == "vector":
                    vector_dimension = field.params.get("dim")
                    break
            
            print(f"Collection '{collection_name}'의 vector 차원: {vector_dimension}")
            
            # PyMilvus 형식에 맞게 데이터 준비 (리스트의 리스트)
            insert_data = []
            
            # 각 필드별로 데이터 수집
            for field in collection.schema.fields:
                field_name = field.name
                if field_name == "id":
                    continue  # id는 auto_id이므로 제외
                
                field_values = []
                for item in data:
                    if field_name == "vector":
                        if "text" in item:
                            vector = await self.text_to_vector(item["text"], target_dimension=vector_dimension)
                            print(f"텍스트 '{item['text'][:50]}...' -> 벡터 차원: {len(vector)}")
                            field_values.append(vector)
                        elif "vector" in item:
                            field_values.append(item["vector"])
                    else:
                        field_values.append(item.get(field_name, ""))
                
                insert_data.append(field_values)
            
            # 벡터 데이터 검증
            if insert_data:
                vectors = insert_data[0]  # 첫 번째 필드가 vector라고 가정
                print(f"삽입할 벡터 개수: {len(vectors)}")
                for i, vec in enumerate(vectors):
                    print(f"벡터 {i+1} 차원: {len(vec)}, 타입: {type(vec)}")
                    if not isinstance(vec, list):
                        raise ValueError(f"벡터 {i+1}이 리스트가 아닙니다: {type(vec)}")
                    if len(vec) != vector_dimension:
                        raise ValueError(f"벡터 {i+1} 차원 불일치: {len(vec)} != {vector_dimension}")
            
            print(f"삽입할 데이터 구조: {len(insert_data)}개 필드")
            collection.insert(insert_data)
            collection.flush()
            
            return {"success": True, "message": f"{len(data)}개의 벡터 삽입 완료"}
        except Exception as e:
            return {"success": False, "message": f"벡터 삽입 실패: {e}"}
    
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