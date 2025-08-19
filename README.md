# Milvus Vector DB CRUD API

FastAPI를 사용하여 구현된 Milvus Vector Database CRUD API입니다. Streamlit 기반의 웹 UI도 제공됩니다.

## 🚀 빠른 시작

### 사전 요구사항
- Docker
- Docker Compose

### 1. 저장소 클론
```bash
git clone https://github.com/woongiiit/Milvus-vector-DB-CRUD-API.git
cd Milvus-vector-DB-CRUD-API
```

### 2. 애플리케이션 실행
```bash
docker-compose up -d
```

### 3. 접속
- **FastAPI 문서**: http://localhost:8000/docs
- **Streamlit UI**: 로컬에서 `streamlit run streamlit_app.py` 실행 후 http://localhost:8501
- **MinIO 콘솔**: http://localhost:9001 (minioadmin/minioadmin)

## 주요 기능

- **컬렉션 관리**: 생성, 삭제, 조회
- **벡터 데이터 관리**: 삽입, 삭제, 조회, 검색
- **다양한 유사도 검색 알고리즘 지원**: L2, IP, COSINE
- **Bulk 작업 지원**: 대량 데이터 삽입/삭제
- **Transformer 기반 텍스트 벡터화**: sentence-transformers 사용
- **비동기 처리**: httpx를 사용한 비동기 통신

## 기술 스택

- **Python**: 3.10.0
- **FastAPI**: 0.104.1
- **Milvus**: 2.3.4
- **Transformers**: 4.35.2
- **PyTorch**: 2.1.1
- **Docker**: 컨테이너화된 배포

## 프로젝트 구조

```
Milvus-vector-DB-CRUD-API/
├── main.py                 # FastAPI 메인 애플리케이션
├── streamlit_app.py        # Streamlit UI
├── requirements.txt        # Python 의존성
├── Dockerfile             # FastAPI 애플리케이션 컨테이너
├── docker-compose.yml     # 전체 서비스 오케스트레이션
├── README.md              # 프로젝트 문서
├── postman_examples.json  # API 테스트 예제
├── Routers/               # API 라우터
│   ├── __init__.py
│   ├── collection_router.py
│   └── vector_router.py
└── Services/              # 비즈니스 로직
    ├── __init__.py
    └── milvus_service.py
```

## API 엔드포인트

### 컬렉션 관리
- `POST /collection/create` - 컬렉션 생성
- `POST /collection/delete` - 컬렉션 삭제
- `GET /collection/collections` - 모든 컬렉션 조회
- `GET /collection/info/{collection_name}` - 컬렉션 정보 조회 (metric_type 포함)

### 벡터 데이터 관리
- `POST /vector/insert` - 벡터 데이터 삽입
- `POST /vector/delete` - 벡터 데이터 삭제
- `GET /vector/vectors` - 벡터 데이터 조회
- `POST /vector/search` - 벡터 검색

## Streamlit UI 기능

### 📚 컬렉션 관리
- 컬렉션 생성 (차원, 거리 측정 방식, 스키마 필드 설정)
- 컬렉션 삭제
- 컬렉션 목록 조회 및 상세 정보 확인

### 🔍 벡터 검색
- 텍스트 기반 벡터 검색
- 검색 결과 개수 및 거리 측정 방식 설정
- 검색 결과 상세 표시

### 📝 벡터 관리
- 벡터 데이터 삽입 (JSON 형식)
- 벡터 데이터 삭제 (ID 기반)

### 📊 데이터 조회
- 컬렉션별 벡터 데이터 조회
- 조회 개수 설정
- 벡터 데이터 상세 표시

## 설치 및 실행

### 1. Docker Compose로 전체 서비스 실행

```bash
# 전체 서비스 시작 (Milvus + FastAPI)
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

### 2. 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# FastAPI 서버 실행 (Milvus가 별도로 실행되어야 함)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Streamlit UI 실행

```bash
# Streamlit UI 실행 (FastAPI 서버가 실행된 상태에서)
streamlit run streamlit_app.py --server.port 8501
```

웹 브라우저에서 `http://localhost:8501`로 접속하여 사용할 수 있습니다.

## API 사용 예제

### 1. 컬렉션 생성

```bash
# L2 거리 계산 방식으로 컬렉션 생성
curl -X POST "http://localhost:8000/collection/create" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents_l2",
    "dimension": 384,
    "metric_type": "L2",
    "schema_fields": [
      {"name": "title", "type": "VARCHAR", "max_length": 100},
      {"name": "content", "type": "VARCHAR", "max_length": 1000},
      {"name": "category", "type": "VARCHAR", "max_length": 50}
    ]
  }'

# 코사인 유사도 방식으로 컬렉션 생성
curl -X POST "http://localhost:8000/collection/create" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents_cosine",
    "dimension": 384,
    "metric_type": "COSINE",
    "schema_fields": [
      {"name": "title", "type": "VARCHAR", "max_length": 100},
      {"name": "content", "type": "VARCHAR", "max_length": 1000},
      {"name": "category", "type": "VARCHAR", "max_length": 50}
    ]
  }'
```

### 2. 벡터 데이터 삽입

```bash
curl -X POST "http://localhost:8000/vector/insert" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "data": [
      {
        "text": "인공지능과 머신러닝에 대한 문서입니다.",
        "title": "AI 문서",
        "content": "이 문서는 인공지능과 머신러닝에 대한 내용을 담고 있습니다.",
        "category": "AI"
      },
      {
        "text": "데이터베이스 시스템에 대한 문서입니다.",
        "title": "DB 문서",
        "content": "이 문서는 데이터베이스 시스템에 대한 내용을 담고 있습니다.",
        "category": "Database"
      }
    ]
  }'
```

### 3. 벡터 검색

```bash
# L2 거리 계산 방식으로 검색
curl -X POST "http://localhost:8000/vector/search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents_l2",
    "query_text": "인공지능에 대해 알려주세요",
    "search_params": {
      "metric_type": "L2",
      "params": {"nprobe": 10}
    },
    "limit": 5
  }'

# 코사인 유사도 방식으로 검색
curl -X POST "http://localhost:8000/vector/search" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents_cosine",
    "query_text": "데이터베이스 시스템",
    "search_params": {
      "metric_type": "COSINE",
      "params": {"nprobe": 10}
    },
    "limit": 3
  }'
```

### 4. 벡터 데이터 조회

```bash
curl -X GET "http://localhost:8000/vector/vectors?collection_name=documents&limit=10"
```

### 5. 벡터 데이터 삭제

```bash
curl -X POST "http://localhost:8000/vector/delete" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "documents",
    "ids": [1, 2, 3]
  }'
```

## 검색 알고리즘 옵션

### Metric Types
- `L2`: 유클리드 거리 (기본값)
- `IP`: 내적 (Inner Product)
- `COSINE`: 코사인 유사도

### Search Parameters
- `nprobe`: 검색할 클러스터 수 (기본값: 10)
- `ef`: HNSW 인덱스의 탐색 깊이
- `search_k`: 검색할 벡터 수

## Postman 테스트

1. **Postman Collection 생성**
2. **환경 변수 설정**:
   - `base_url`: `http://localhost:8000`
3. **API 요청 예제**:
   - Collection 생성 → 벡터 삽입 → 벡터 검색 → 벡터 삭제 순서로 테스트

## 주의사항

1. **Milvus 연결**: 애플리케이션 시작 시 Milvus 서버가 실행되어야 합니다.
2. **메모리 사용량**: Transformer 모델 로딩으로 인한 메모리 사용량 증가
3. **벡터 차원**: 컬렉션 생성 시 설정한 차원과 실제 벡터 차원이 일치해야 합니다.
4. **인덱스**: 컬렉션 생성 시 자동으로 인덱스가 생성됩니다.
5. **Metric Type 일치**: 컬렉션 생성 시 설정한 거리 계산 방식과 검색 시 사용하는 방식이 일치해야 합니다.
6. **에러 처리**: Metric type 불일치 시 명확한 에러 메시지와 함께 컬렉션 정보를 제공합니다.

## 개발 환경

- **Python**: 3.10.0
- **OS**: Windows 10
- **Docker**: 컨테이너화된 배포
- **API 문서**: `http://localhost:8000/docs` (Swagger UI)

## 기여하기

1. 이 저장소를 Fork합니다
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 Push합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 