import streamlit as st
import requests
import json
from typing import List, Dict, Any

# API 기본 URL
API_BASE_URL = "http://localhost:8000"

def make_api_request(method: str, endpoint: str, data: Dict = None, params: Dict = None):
    """API 요청을 보내는 헬퍼 함수"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    st.set_page_config(
        page_title="Milvus Vector DB 관리자",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Milvus Vector DB 관리자")
    st.markdown("---")
    
    # 사이드바 - API 상태 확인
    with st.sidebar:
        st.header("📊 API 상태")
        health_response = make_api_request("GET", "/health")
        if health_response["success"]:
            st.success("✅ API 서버 연결됨")
        else:
            st.error("❌ API 서버 연결 실패")
            st.stop()
    
    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["📚 컬렉션 관리", "🔍 벡터 검색", "📝 벡터 관리", "📊 데이터 조회"])
    
    # 탭 1: 컬렉션 관리
    with tab1:
        st.header("📚 컬렉션 관리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("컬렉션 생성")
            with st.form("create_collection"):
                collection_name = st.text_input("컬렉션 이름", key="create_name")
                dimension = st.number_input("벡터 차원", min_value=1, value=768, key="create_dim")
                metric_type = st.selectbox("거리 측정 방식", ["L2", "IP", "COSINE"], key="create_metric")
                
                # 스키마 필드 추가
                st.write("스키마 필드 (선택사항)")
                schema_fields = []
                num_fields = st.number_input("필드 개수", min_value=0, max_value=10, value=0, key="schema_count")
                
                for i in range(num_fields):
                    with st.expander(f"필드 {i+1}"):
                        field_name = st.text_input(f"필드 이름 {i+1}", key=f"field_name_{i}")
                        field_type = st.selectbox(f"필드 타입 {i+1}", ["VARCHAR", "INT64", "FLOAT"], key=f"field_type_{i}")
                        max_length = st.number_input(f"최대 길이 {i+1}", min_value=1, value=65535, key=f"field_length_{i}")
                        
                        if field_name:
                            schema_fields.append({
                                "name": field_name,
                                "type": field_type,
                                "max_length": max_length
                            })
                
                submit_create = st.form_submit_button("컬렉션 생성")
                
                if submit_create and collection_name:
                    data = {
                        "collection_name": collection_name,
                        "dimension": dimension,
                        "schema_fields": schema_fields,
                        "metric_type": metric_type
                    }
                    
                    result = make_api_request("POST", "/collection/create", data)
                    if result["success"]:
                        st.success("✅ 컬렉션이 성공적으로 생성되었습니다!")
                    else:
                        st.error(f"❌ 생성 실패: {result['error']}")
        
        with col2:
            st.subheader("컬렉션 삭제")
            with st.form("delete_collection"):
                delete_collection_name = st.text_input("삭제할 컬렉션 이름", key="delete_name")
                submit_delete = st.form_submit_button("컬렉션 삭제")
                
                if submit_delete and delete_collection_name:
                    data = {"collection_name": delete_collection_name}
                    result = make_api_request("POST", "/collection/delete", data)
                    if result["success"]:
                        st.success("✅ 컬렉션이 성공적으로 삭제되었습니다!")
                    else:
                        st.error(f"❌ 삭제 실패: {result['error']}")
        
        # 컬렉션 목록 조회
        st.subheader("📋 컬렉션 목록")
        if st.button("컬렉션 목록 새로고침"):
            result = make_api_request("GET", "/collection/collections")
            if result["success"]:
                collections = result["data"]["collections"]
                if collections:
                    for collection in collections:
                        with st.expander(f"📁 {collection}"):
                            # 컬렉션 정보 조회
                            info_result = make_api_request("GET", f"/collection/info/{collection}")
                            if info_result["success"]:
                                info = info_result["data"]["collection_info"]
                                st.json(info)
                            else:
                                st.error("컬렉션 정보 조회 실패")
                else:
                    st.info("등록된 컬렉션이 없습니다.")
            else:
                st.error(f"컬렉션 목록 조회 실패: {result['error']}")
    
    # 탭 2: 벡터 검색
    with tab2:
        st.header("🔍 벡터 검색")
        
        with st.form("vector_search"):
            search_collection = st.text_input("검색할 컬렉션 이름", key="search_collection")
            query_text = st.text_area("검색할 텍스트", key="search_text")
            search_limit = st.number_input("검색 결과 개수", min_value=1, max_value=100, value=10, key="search_limit")
            metric_type = st.selectbox("거리 측정 방식", ["L2", "IP", "COSINE"], key="search_metric")
            
            submit_search = st.form_submit_button("🔍 검색")
            
            if submit_search and search_collection and query_text:
                data = {
                    "collection_name": search_collection,
                    "query_text": query_text,
                    "search_params": {
                        "metric_type": metric_type,
                        "params": {"nprobe": 10}
                    },
                    "limit": search_limit
                }
                
                result = make_api_request("POST", "/vector/search", data)
                if result["success"]:
                    search_results = result["data"]["results"]
                    st.success(f"✅ 검색 완료! {len(search_results)}개 결과")
                    
                    for i, result_item in enumerate(search_results):
                        with st.expander(f"결과 {i+1} (거리: {result_item.get('distance', 'N/A')})"):
                            st.json(result_item)
                else:
                    st.error(f"❌ 검색 실패: {result['error']}")
    
    # 탭 3: 벡터 관리
    with tab3:
        st.header("📝 벡터 관리")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("벡터 삽입")
            with st.form("insert_vectors"):
                insert_collection = st.text_input("컬렉션 이름", key="insert_collection")
                
                # 벡터 데이터 입력
                st.write("벡터 데이터 (JSON 형식)")
                vector_data_json = st.text_area(
                    "벡터 데이터 JSON",
                    value='[{"text": "샘플 텍스트", "vector": [0.1, 0.2, 0.3]}]',
                    key="insert_data"
                )
                
                submit_insert = st.form_submit_button("벡터 삽입")
                
                if submit_insert and insert_collection:
                    try:
                        vector_data = json.loads(vector_data_json)
                        data = {
                            "collection_name": insert_collection,
                            "data": vector_data
                        }
                        
                        result = make_api_request("POST", "/vector/insert", data)
                        if result["success"]:
                            st.success("✅ 벡터가 성공적으로 삽입되었습니다!")
                        else:
                            st.error(f"❌ 삽입 실패: {result['error']}")
                    except json.JSONDecodeError:
                        st.error("❌ JSON 형식이 올바르지 않습니다.")
        
        with col2:
            st.subheader("벡터 삭제")
            with st.form("delete_vectors"):
                delete_vectors_collection = st.text_input("컬렉션 이름", key="delete_vectors_collection")
                vector_ids = st.text_input("삭제할 벡터 ID들 (쉼표로 구분)", key="delete_ids")
                
                submit_delete_vectors = st.form_submit_button("벡터 삭제")
                
                if submit_delete_vectors and delete_vectors_collection and vector_ids:
                    try:
                        ids = [int(id.strip()) for id in vector_ids.split(",")]
                        data = {
                            "collection_name": delete_vectors_collection,
                            "ids": ids
                        }
                        
                        result = make_api_request("POST", "/vector/delete", data)
                        if result["success"]:
                            st.success("✅ 벡터가 성공적으로 삭제되었습니다!")
                        else:
                            st.error(f"❌ 삭제 실패: {result['error']}")
                    except ValueError:
                        st.error("❌ ID 형식이 올바르지 않습니다. 숫자를 쉼표로 구분해주세요.")
    
    # 탭 4: 데이터 조회
    with tab4:
        st.header("📊 데이터 조회")
        
        with st.form("get_vectors"):
            get_collection = st.text_input("조회할 컬렉션 이름", key="get_collection")
            get_limit = st.number_input("조회할 벡터 개수", min_value=1, max_value=1000, value=100, key="get_limit")
            
            submit_get = st.form_submit_button("벡터 조회")
            
            if submit_get and get_collection:
                params = {
                    "collection_name": get_collection,
                    "limit": get_limit
                }
                
                result = make_api_request("GET", "/vector/vectors", params=params)
                if result["success"]:
                    vectors = result["data"]["vectors"]
                    st.success(f"✅ {len(vectors)}개의 벡터를 조회했습니다!")
                    
                    # 벡터 데이터 표시
                    for i, vector in enumerate(vectors):
                        with st.expander(f"벡터 {i+1} (ID: {vector.get('id', 'N/A')})"):
                            st.json(vector)
                else:
                    st.error(f"❌ 조회 실패: {result['error']}")

if __name__ == "__main__":
    main() 