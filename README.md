# SmartRAG - Intelligent Retrieval-Augmented Generation Service

## 🎯 프로젝트 개요
SmartRAG는 대규모 문서를 기반으로 지능형 질의응답을 제공하는 RAG(Retrieval-Augmented Generation) 서비스입니다. 벡터 데이터베이스와 LLM을 결합하여 정확하고 빠른 정보 검색 및 답변 생성을 지원합니다.

## 🚀 핵심 기능
- **문서 수집 및 처리**: 다양한 형식의 문서 자동 수집, 정제, 청킹
- **벡터 임베딩**: Sentence-Transformers를 활용한 고성능 임베딩 생성
- **의미 기반 검색**: FAISS/pgvector를 통한 빠른 유사도 검색
- **스트리밍 응답**: 실시간 스트리밍으로 빠른 사용자 경험 제공
- **캐싱 최적화**: Redis 기반 지능형 캐싱으로 응답 속도 향상

## 🛠️ 기술 스택
### Backend
- **Language**: Python 3.11
- **Framework**: FastAPI
- **LLM Serving**: vLLM
- **Embedding**: sentence-transformers

### Data & Storage
- **Vector DB**: FAISS, pgvector
- **Cache**: Redis
- **Object Storage**: S3/MinIO

### Infrastructure
- **Container**: Docker (multi-stage build)
- **Orchestration**: Kubernetes (HPA, Canary deployment)
- **Monitoring**: OpenTelemetry, Prometheus, Grafana

## 📊 성능 목표
- **응답 시간**: p95 < 1.2초
- **캐시 히트율**: > 60%
- **가용성**: 99.5% SLO
- **동시 처리**: 100+ 요청/초

## 🏗️ 아키텍처
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI     │────▶│    vLLM     │
└─────────────┘     │   Gateway    │     │   Server    │
                    └──────────────┘     └─────────────┘
                           │                     │
                    ┌──────▼──────┐       ┌─────▼──────┐
                    │    Redis    │       │  Vector    │
                    │    Cache    │       │     DB     │
                    └─────────────┘       └────────────┘
```

## 📁 프로젝트 구조
```
SmartRAG/
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── core/          # Core business logic
│   ├── embeddings/    # Embedding generation
│   ├── retrieval/     # Vector search logic
│   └── llm/           # LLM integration
├── k8s/               # Kubernetes manifests
├── docker/            # Dockerfile and configs
├── tests/             # Unit and integration tests
└── monitoring/        # Dashboards and alerts
```

## 🚦 API Endpoints
- `POST /ingest` - 문서 수집 및 인덱싱
- `POST /query` - 질의응답 요청
- `GET /health` - 서비스 상태 확인
- `GET /metrics` - Prometheus 메트릭

## 🔧 설치 및 실행
```bash
# 로컬 개발 환경
docker-compose up -d

# Kubernetes 배포
kubectl apply -f k8s/

# 테스트 실행
pytest tests/ --cov=src
```

## 📈 개발 로드맵
- [x] Week 1: 데이터 파이프라인 구축
- [x] Week 2: API 개발 및 LLM 서빙
- [x] Week 3: Kubernetes 배포 및 모니터링
- [ ] 성능 최적화 및 튜닝
- [ ] A/B 테스팅 프레임워크
- [ ] 멀티테넌시 지원

## 📚 문서
- [API Documentation](./docs/api.md)
- [Architecture Guide](./docs/architecture.md)
- [Deployment Guide](./docs/deployment.md)
- [Performance Tuning](./docs/performance.md)

## 📄 라이선스
MIT License`c