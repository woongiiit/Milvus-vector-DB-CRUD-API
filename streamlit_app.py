import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any

# 페이지 설정
st.set_page_config(
    page_title="Milvus Vector DB 관리자",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 (모던한 블랙 톤)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .stButton > button {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #374151;
        border-color: #4b5563;
    }
    .stSelectbox > div > div {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
    .stTextArea > div > div > textarea {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
    .stNumberInput > div > div > input {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
    .stMetric {
        background-color: #1f2937;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #374151;
    }
    .success-message {
        background-color: #065f46;
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #047857;
        margin: 10px 0;
    }
    .error-message {
        background-color: #991b1b;
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dc2626;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #92400e;
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #d97706;
        margin: 10px 0;
    }
    .info-message {
        background-color: #1e40af;
        color: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #3b82f6;
        margin: 10px 0;
    }
    .stExpander {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    .stTabs > div > div > div > div {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
    }
    .stTabs > div > div > div > div[data-baseweb="tab"] {
        background-color: #374151;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# API 기본 URL
API_BASE_URL = "http://localhost:8000"

class MilvusAPI:
    """Milvus API 클라이언트"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_collections(self) -> Dict[str, Any]:
        """컬렉션 목록 조회"""
        try:
            response = self.session.get(f"{self.base_url}/collection/collections", timeout=10)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            response = self.session.get(f"{self.base_url}/collection/info/{collection_name}", timeout=10)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_collection(self, collection_data: Dict[str, Any]) -> Dict[str, Any]:
        """컬렉션 생성"""
        try:
            response = self.session.post(f"{self.base_url}/collection/create", 
                                       json=collection_data, timeout=15)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 삭제"""
        try:
            response = self.session.post(f"{self.base_url}/collection/delete", 
                                       json={"collection_name": collection_name}, timeout=15)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def bulk_delete_collections(self, collection_names: List[str]) -> Dict[str, Any]:
        """여러 컬렉션 삭제"""
        try:
            response = self.session.post(f"{self.base_url}/collection/bulk_delete", 
                                       json={"collection_names": collection_names}, timeout=20)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def insert_vectors(self, collection_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """벡터 데이터 삽입"""
        try:
            response = self.session.post(f"{self.base_url}/vector/insert", 
                                       json={"collection_name": collection_name, "data": data}, timeout=30)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_vectors(self, collection_name: str, limit: int = 100) -> Dict[str, Any]:
        """벡터 데이터 조회"""
        try:
            response = self.session.get(f"{self.base_url}/vector/vectors", 
                                      params={"collection_name": collection_name, "limit": limit}, timeout=15)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_vectors(self, collection_name: str, ids: List[int]) -> Dict[str, Any]:
        """벡터 데이터 삭제"""
        try:
            response = self.session.post(f"{self.base_url}/vector/delete", 
                                       json={"collection_name": collection_name, "ids": ids}, timeout=15)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_vectors(self, collection_name: str, query_text: str, 
                      metric_type: str, limit: int = 10) -> Dict[str, Any]:
        """벡터 검색"""
        try:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
            
            request_data = {
                "collection_name": collection_name,
                "query_text": query_text,
                "search_params": search_params,
                "limit": limit
            }
            
            response = self.session.post(f"{self.base_url}/vector/search", 
                                       json=request_data, timeout=20)
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

def show_message(message: str, message_type: str = "info"):
    """메시지 표시 (타입별 스타일 적용)"""
    if message_type == "success":
        st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
    elif message_type == "error":
        st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
    elif message_type == "warning":
        st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-message">{message}</div>', unsafe_allow_html=True)

def dashboard_page():
    """대시보드 페이지"""
    st.title("🏠 Milvus Vector DB 대시보드")
    
    api = MilvusAPI(API_BASE_URL)
    
    # API 상태 확인
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ 백엔드 API 서버가 정상적으로 실행 중입니다.")
        else:
            st.error("❌ 백엔드 API 서버에 연결할 수 없습니다.")
            return
    except Exception as e:
        st.error(f"❌ 백엔드 API 서버 연결 실패: {str(e)}")
        return
    
    # 컬렉션 통계
    collections_result = api.get_collections()
    if collections_result["success"]:
        collections_data = collections_result["data"]
        if collections_data["status"] == "success" and collections_data["collections"]:
            collections = collections_data["collections"]
            
            # 통계 정보
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 컬렉션 수", len(collections))
            with col2:
                total_vectors = sum(col.get("entity_count", 0) for col in collections)
                st.metric("총 벡터 수", total_vectors)
            with col3:
                avg_dimension = sum(col.get("dimension", 0) for col in collections) / len(collections) if collections else 0
                st.metric("평균 벡터 차원", f"{avg_dimension:.0f}")
            with col4:
                st.metric("시스템 상태", "정상")
            
            # 컬렉션 목록
            st.subheader("📚 컬렉션 목록")
            collections_df = pd.DataFrame(collections)
            st.dataframe(collections_df, use_container_width=True)
        else:
            st.info("📭 아직 생성된 컬렉션이 없습니다.")
    else:
        st.error(f"❌ 컬렉션 정보 조회 실패: {collections_result['error']}")

def collection_management_page():
    """컬렉션 관리 페이지"""
    st.title("🗂️ 컬렉션 관리")
    
    api = MilvusAPI(API_BASE_URL)
    
    # 새 컬렉션 생성
    st.subheader("🚀 새 컬렉션 생성")
    
    col1, col2 = st.columns(2)
    with col1:
        collection_name = st.text_input("컬렉션 이름", placeholder="예: documents")
        dimension = st.number_input("벡터 차원", min_value=1, max_value=4096, value=384, step=1,
                                  help="벡터의 차원 수 (예: 384, 768, 1024)")
    with col2:
        metric_type = st.selectbox("메트릭 타입", 
                                 ["COSINE", "L2", "IP"],
                                 help="COSINE: 코사인 유사도 (텍스트 검색 최적), L2: 유클리드 거리, IP: 내적")
        index_type = st.selectbox("인덱스 타입", 
                                ["IVF_FLAT", "HNSW", "IVF_SQ8", "FLAT"],
                                help="IVF_FLAT: 균형잡힌 성능, HNSW: 빠른 검색, FLAT: 정확한 검색")
    
    # 컬렉션 생성 버튼
    if st.button("🚀 컬렉션 생성", type="primary"):
        if collection_name:
            collection_data = {
                "collection_name": collection_name,
                "dimension": dimension,
                "metric_type": metric_type,
                "index_type": index_type
            }
            
            with st.spinner("컬렉션 생성 중..."):
                result = api.create_collection(collection_data)
            
            if result["success"]:
                show_message(f"✅ 컬렉션 '{collection_name}'이 성공적으로 생성되었습니다!", "success")
                st.balloons()
                st.rerun()
            else:
                show_message(f"❌ 컬렉션 생성 실패: {result['error']}", "error")
        else:
            show_message("⚠️ 컬렉션 이름을 입력해주세요.", "warning")
    
    # 기존 컬렉션 관리
    st.subheader("🗂️ 기존 컬렉션")
    
    collections_result = api.get_collections()
    if collections_result["success"]:
        collections_data = collections_result["data"]
        if collections_data["status"] == "success" and collections_data["collections"]:
            collections = collections_data["collections"]
            
            # 컬렉션 목록을 체크박스와 함께 표시
            st.write("📋 컬렉션 목록 (삭제할 컬렉션을 체크하세요)")
            
            # 체크박스와 컬렉션 정보를 함께 표시
            selected_collections = []
            
            # 각 컬렉션을 개별적으로 표시하여 체크박스와 정보를 연결
            for collection in collections:
                collection_name = collection['name']
                
                # 컬렉션 정보를 카드 형태로 표시
                with st.container():
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 1, 1.5, 1.5, 1, 1])
                    
                    with col1:
                        # 체크박스
                        is_selected = st.checkbox("", key=f"select_{collection_name}")
                        if is_selected:
                            selected_collections.append(collection_name)
                    
                    with col2:
                        st.write(f"**{collection_name}**")
                    
                    with col3:
                        st.write(f"차원: {collection.get('dimension', 'N/A')}")
                    
                    with col4:
                        st.write(f"메트릭: {collection.get('metric_type', 'N/A')}")
                    
                    with col5:
                        st.write(f"벡터: {collection.get('entity_count', 0)}개")
                    
                    with col6:
                        st.write(f"인덱스: {collection.get('index_type', 'N/A')}")
                    
                    with col7:
                        st.write("")
            
            # 일괄 삭제 버튼 (컬렉션 선택 후)
            if selected_collections:
                st.write(f"🗑️ **선택된 컬렉션: {', '.join(selected_collections)}**")
                
                # 삭제 확인 체크박스
                confirm_delete = st.checkbox("삭제를 확인합니다", key="confirm_bulk_delete")
                
                if confirm_delete:
                    if st.button("🗑️ 선택된 컬렉션 삭제", type="primary"):
                        with st.spinner("컬렉션 삭제 중..."):
                            result = api.bulk_delete_collections(selected_collections)
                        
                        if result["success"]:
                            show_message(f"✅ {len(selected_collections)}개의 컬렉션이 성공적으로 삭제되었습니다!", "success")
                            st.rerun()
                        else:
                            show_message(f"❌ 컬렉션 삭제 실패: {result['error']}", "error")
                else:
                    st.info("삭제할 컬렉션이 있으면 위의 체크박스를 선택해주세요.")
            else:
                st.info("삭제할 컬렉션이 있으면 위의 체크박스를 선택해주세요.")
        else:
            st.info("📭 아직 생성된 컬렉션이 없습니다.")
    else:
        st.error(f"❌ 컬렉션 정보 조회 실패: {collections_result['error']}")

def vector_data_page():
    """벡터 데이터 관리 페이지"""
    st.title("📊 벡터 데이터 관리")
    
    api = MilvusAPI(API_BASE_URL)
    
    # 컬렉션 선택
    collections_result = api.get_collections()
    if not collections_result["success"]:
        show_message(f"❌ 컬렉션 정보 조회 실패: {collections_result['error']}", "error")
        return
    
    collections_data = collections_result["data"]
    if collections_data["status"] != "success" or not collections_data["collections"]:
        show_message("📭 사용 가능한 컬렉션이 없습니다. 먼저 컬렉션을 생성해주세요.", "warning")
        return
    
    collection_names = [col["name"] for col in collections_data["collections"]]
    selected_collection = st.selectbox("📁 컬렉션 선택", collection_names)
    
    if not selected_collection:
        return
    
    # 컬렉션 정보 표시
    collection_info_result = api.get_collection_info(selected_collection)
    if collection_info_result["success"]:
        info_data = collection_info_result["data"]
        
        # metric_type과 dimension 추출
        metric_type = info_data.get('collection_info', {}).get('metric_type', 'UNKNOWN')
        dimension = info_data.get('collection_info', {}).get('dimension', 'N/A')
        
        # 컬렉션 기본 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("컬렉션명", selected_collection)
        with col2:
            st.metric("벡터 차원", dimension)
        with col3:
            st.metric("메트릭 타입", metric_type)
        
        # 시스템 필드 정보 표시
        st.info("📋 **시스템 필드만 사용 가능합니다** (id, vector)")
    
    # 데이터 삽입
    with st.expander("📥 벡터 데이터 삽입", expanded=True):
        tab1, tab2 = st.tabs(["📝 텍스트 입력", "🔢 벡터 직접 입력"])
        
        with tab1:
            st.subheader("텍스트 기반 데이터 삽입")
            
            # 샘플 데이터
            if st.button("🎯 샘플 데이터 삽입"):
                st.info("📋 **시스템 필드만 사용합니다** (id, vector)")
                
                # 시스템 필드만 사용 (텍스트만)
                sample_data = [
                    {"text": "인공지능과 머신러닝에 대한 문서입니다."},
                    {"text": "데이터베이스 시스템에 대한 문서입니다."},
                    {"text": "웹 개발과 프론트엔드 기술에 대한 문서입니다."}
                ]
                st.write("시스템 필드만 사용하여 텍스트만 포함된 샘플 데이터 생성")
                
                with st.spinner("샘플 데이터 삽입 중..."):
                    result = api.insert_vectors(selected_collection, sample_data)
                
                if result["success"]:
                    show_message(f"✅ 샘플 데이터 {len(sample_data)}개가 성공적으로 삽입되었습니다!", "success")
                else:
                    show_message(f"❌ 샘플 데이터 삽입 실패: {result['error']}", "error")
            
            # 커스텀 데이터 입력
            st.subheader("커스텀 데이터 입력")
            
            text_input = st.text_area("텍스트 내용", placeholder="벡터로 변환할 텍스트를 입력하세요...")
            
            st.info("📋 **시스템 필드만 사용합니다** (id, vector)")
            if st.button("📤 데이터 삽입", type="primary"):
                if text_input:
                    # 데이터 구성 (시스템 필드만 사용)
                    custom_data = [{"text": text_input}]
                    
                    with st.spinner("데이터 삽입 중..."):
                        result = api.insert_vectors(selected_collection, custom_data)
                    
                    if result["success"]:
                        show_message("✅ 커스텀 데이터가 성공적으로 삽입되었습니다!", "success")
                        st.rerun()
                    else:
                        show_message(f"❌ 데이터 삽입 실패: {result['error']}", "error")
                else:
                    show_message("⚠️ 텍스트 내용을 입력해주세요.", "warning")
        
        with tab2:
            st.subheader("벡터 직접 입력")
            st.info("⚠️ 주의: 벡터 차원이 컬렉션 차원과 일치해야 합니다.")
            
            vector_input = st.text_area("벡터 데이터 (JSON 형식)", 
                                      placeholder='[0.1, 0.2, 0.3, ...]',
                                      help="숫자 배열 형태로 입력하세요")
            
            if st.button("📤 벡터 삽입"):
                if vector_input:
                    try:
                        vector_data = json.loads(vector_input)
                        if isinstance(vector_data, list) and all(isinstance(x, (int, float)) for x in vector_data):
                            data = [{"vector": vector_data}]
                            
                            with st.spinner("벡터 삽입 중..."):
                                result = api.insert_vectors(selected_collection, data)
                            
                            if result["success"]:
                                show_message("✅ 벡터가 성공적으로 삽입되었습니다!", "success")
                                st.rerun()
                            else:
                                show_message(f"❌ 벡터 삽입 실패: {result['error']}", "error")
                        else:
                            show_message("❌ 올바른 벡터 형식이 아닙니다.", "error")
                    except json.JSONDecodeError:
                        show_message("❌ JSON 형식이 올바르지 않습니다.", "error")
                else:
                    show_message("⚠️ 벡터 데이터를 입력해주세요.", "warning")
    
    # 데이터 조회 및 삭제
    st.subheader("📋 데이터 조회")
    
    vectors_result = api.get_vectors(selected_collection)
    if vectors_result["success"]:
        vectors_data = vectors_result["data"]
        if vectors_data["status"] == "success" and vectors_data["vectors"]:
            vectors = vectors_data["vectors"]
            
            # 데이터프레임으로 변환
            df_data = []
            for vector in vectors:
                row = {
                    "ID": vector.get("id", "N/A"),
                    "벡터 차원": len(vector.get("vector", [])),
                    "삽입 시간": vector.get("insert_time", "N/A")
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # 데이터 삭제
            st.subheader("🗑️ 데이터 삭제")
            delete_ids = st.multiselect("삭제할 데이터 ID 선택", 
                                      options=[row["ID"] for row in df_data if row["ID"] != "N/A"],
                                      help="삭제할 데이터의 ID를 선택하세요")
            
            if delete_ids and st.button("🗑️ 선택된 데이터 삭제", type="primary"):
                with st.spinner("데이터 삭제 중..."):
                    result = api.delete_vectors(selected_collection, delete_ids)
                
                if result["success"]:
                    show_message(f"✅ {len(delete_ids)}개의 데이터가 삭제되었습니다.", "success")
                    st.rerun()
                else:
                    show_message(f"❌ 데이터 삭제 실패: {result['error']}", "error")
        else:
            show_message("📭 삽입된 데이터가 없습니다.", "info")
    else:
        show_message(f"❌ 데이터 조회 실패: {vectors_result['error']}", "error")

def vector_search_page():
    """벡터 검색 페이지"""
    st.title("🔍 벡터 검색")
    
    api = MilvusAPI(API_BASE_URL)
    
    # 컬렉션 선택
    collections_result = api.get_collections()
    if not collections_result["success"]:
        show_message(f"❌ 컬렉션 정보 조회 실패: {collections_result['error']}", "error")
        return
    
    collections_data = collections_result["data"]
    if collections_data["status"] != "success" or not collections_data["collections"]:
        show_message("📭 사용 가능한 컬렉션이 없습니다. 먼저 컬렉션을 생성해주세요.", "warning")
        return
    
    collection_names = [col["name"] for col in collections_data["collections"]]
    selected_collection = st.selectbox("📁 검색할 컬렉션 선택", collection_names)
    
    if not selected_collection:
        return
    
    # 컬렉션 정보 표시
    collection_info_result = api.get_collection_info(selected_collection)
    if collection_info_result["success"]:
        info_data = collection_info_result["data"]["collection_info"]
        metric_type = info_data.get("metric_type", "UNKNOWN")
        st.info(f"선택된 컬렉션: {selected_collection} | 메트릭 타입: {metric_type}")
    
    # 검색 설정
    st.subheader("🔍 검색 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query_text = st.text_area("검색 쿼리", placeholder="검색하고 싶은 텍스트를 입력하세요...", height=100)
    
    with col2:
        metric_type = st.selectbox("메트릭 타입", 
                                 ["COSINE", "L2", "IP"],
                                 help="컬렉션 생성 시 설정한 메트릭 타입과 일치해야 합니다")
        limit = st.number_input("검색 결과 수", min_value=1, max_value=50, value=10, step=1)
    
    with col3:
        nprobe = st.number_input("nprobe 값", min_value=1, max_value=100, value=10, step=1,
                                help="높을수록 정확하지만 느림")
    
    # 검색 실행
    if st.button("🔍 검색 실행", type="primary"):
        if query_text:
            with st.spinner("검색 중..."):
                search_result = api.search_vectors(selected_collection, query_text, metric_type, limit)
            
            if search_result["success"]:
                search_data = search_result["data"]
                
                # 응답 구조 디버깅을 위한 로그
                st.info(f"🔍 API 응답 구조: {list(search_data.keys())}")
                
                # status 키가 있는지 확인하고 안전하게 처리
                if "status" in search_data:
                    if search_data["status"] == "success":
                        # results 키가 있는지 확인
                        if "results" in search_data:
                            results = search_data["results"]
                            st.success(f"✅ 검색 완료! {len(results)}개의 결과를 찾았습니다.")
                            
                            # 검색 결과 표시
                            if results:
                                for i, result in enumerate(results):
                                    with st.expander(f"🔍 결과 {i+1} (유사도: {result.get('distance', 'N/A'):.4f})", expanded=False):
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            st.write(f"**ID:** {result.get('id', 'N/A')}")
                                            st.write(f"**유사도 점수:** {result.get('distance', 'N/A'):.4f}")
                                            if 'score' in result:
                                                st.write(f"**점수:** {result.get('score', 'N/A'):.4f}")
                                        
                                        with col2:
                                            if 'vector' in result:
                                                vector = result['vector']
                                                st.write(f"**벡터 차원:** {len(vector)}")
                                                st.write(f"**벡터 샘플:** {vector[:5]}...")
                        else:
                            st.error("❌ 검색 결과가 올바르지 않습니다. 'results' 키가 없습니다.")
                    else:
                        st.error(f"❌ 검색 실패: {search_data.get('message', '알 수 없는 오류')}")
                else:
                    st.error("❌ 검색 응답에 'status' 키가 없습니다.")
                    st.write("🔍 **디버깅 정보:**")
                    st.write(f"응답 키: {list(search_data.keys())}")
                    st.write(f"전체 응답: {search_data}")
        else:
            show_message("⚠️ 검색 쿼리를 입력해주세요.", "warning")

def main():
    """메인 함수"""
    # 사이드바 - 페이지 선택
    with st.sidebar:
        st.title("🔍 Milvus 관리자")
        st.markdown("---")
        
        # API 상태 확인
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API 연결됨")
            else:
                st.error("❌ API 연결 실패")
                st.stop()
        except Exception as e:
            st.error(f"❌ API 연결 실패: {str(e)}")
            st.stop()
        
        # 페이지 선택
        page = st.selectbox(
            "페이지 선택",
            ["🏠 대시보드", "🗂️ 컬렉션 관리", "📊 벡터 데이터", "🔍 벡터 검색"]
        )
    
    # 페이지 라우팅
    if page == "🏠 대시보드":
        dashboard_page()
    elif page == "🗂️ 컬렉션 관리":
        collection_management_page()
    elif page == "📊 벡터 데이터":
        vector_data_page()
    elif page == "🔍 벡터 검색":
        vector_search_page()

if __name__ == "__main__":
    main()
