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
- **로깅**: Python logging 모듈 (한국 시간대 지원)

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
    ├── milvus_service.py
    └── logger_config.py   # 로거 설정
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
7. **로깅**: 모든 로그는 한국 시간대(KST)로 기록되며, `logs/` 디렉토리에 저장됩니다.

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

## 로깅 시스템

### 로거 설정
이 프로젝트는 Python의 `logging` 모듈을 사용하여 체계적인 로깅을 제공합니다.

#### 주요 기능
- **한국 시간대 지원**: 모든 로그는 한국 표준시(KST)로 기록됩니다.
- **파일 및 콘솔 출력**: 로그는 콘솔과 파일에 동시에 출력됩니다.
- **일별 로테이션**: 로그 파일은 매일 자정에 새로운 파일로 생성되며, 30일간 보관됩니다.
- **로그 레벨**: INFO, WARNING, ERROR 등 다양한 로그 레벨을 지원합니다.
- **사용자 행위 추적**: 사용자의 모든 요청과 결과를 별도로 기록합니다.

#### 로그 파일 위치
```
logs/
├── milvus_service.log          # 시스템 동작 로그 (현재)
├── milvus_service.log.2025-08-18  # 시스템 동작 로그 (어제)
├── user_activity.log           # 사용자 행위 로그 (현재)
├── user_activity.log.2025-08-18  # 사용자 행위 로그 (어제)
└── ...
```

#### 로그 포맷
```
2025-08-19 15:09:36 대한민국 표준시 [INFO] MilvusService: 📋 [CREATE] 컬렉션 생성 시작
2025-08-19 15:09:36 대한민국 표준시 [WARNING] MilvusService: ⚠️ 경고 메시지
2025-08-19 15:09:36 대한민국 표준시 [ERROR] MilvusService: ❌ 오류 메시지
```

#### 사용자 행위 추적 로그
- **👤 [USER_ACTION]**: 사용자 요청 시작
- **✅ [USER_SUCCESS]**: 사용자 요청 성공
- **❌ [USER_FAILURE]**: 사용자 요청 실패
- **⚠️ [USER_PARTIAL]**: 사용자 요청 부분 성공

#### 시스템 동작 로그
- **📋 [CREATE]**: 컬렉션 생성 과정
- **🔍 [SEARCH]**: 벡터 검색 과정
- **📥 [INSERT]**: 벡터 삽입 과정
- **🗑️ [DELETE]**: 컬렉션 삭제 과정
- **📊 [VECTOR]**: 벡터 처리 과정

#### 로거 사용법
```python
from Services.logger_config import get_milvus_logger, get_user_activity_logger

# 시스템 동작 로거
logger = get_milvus_logger()
logger.info("정보 메시지")
logger.warning("경고 메시지")
logger.error("오류 메시지")

# 사용자 행위 추적 로거
user_logger = get_user_activity_logger()
user_logger.info("사용자 행위 기록")
```

#### 로그 예시
```
# 사용자 행위 로그
2025-08-19 15:09:36 KST [INFO] UserActivity: 👤 [USER_ACTION] 벡터 검색 요청 - 컬렉션: test_collection, 쿼리: '안녕하세요', limit: 10
2025-08-19 15:09:37 KST [INFO] UserActivity: ✅ [USER_SUCCESS] 벡터 검색 완료 - 컬렉션: test_collection, 쿼리: '안녕하세요', 결과 개수: 5

# 시스템 동작 로그
2025-08-19 15:09:36 KST [INFO] MilvusService: 🔍 [SEARCH] 검색 시작 - 컬렉션: test_collection, 쿼리: '안녕하세요', limit: 10
2025-08-19 15:09:37 KST [INFO] MilvusService: 🎉 [SEARCH] 검색 완료 성공! - 결과 개수: 5
```

#### 한국 시간대 유틸리티
```python
from Services.logger_config import get_korea_time, format_korea_time

# 현재 한국 시간 가져오기
korea_time = get_korea_time()

# 한국 시간을 포맷팅하여 문자열로 변환
formatted_time = format_korea_time()
```

### 로그 설정 커스터마이징
`Services/logger_config.py` 파일에서 로거 설정을 수정할 수 있습니다:

- **로그 레벨 변경**: `setup_logger()` 함수의 `level` 매개변수 수정
- **로그 보관 기간**: `backupCount` 매개변수 수정 (기본값: 30일)
- **로그 포맷 변경**: `formatter` 설정 수정

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 