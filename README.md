## AI 관상 분석 (Face Reading AI Service)

### 로컬 개발 환경 Setup
1. Python 3.9+ 가상환경 생성
2. 의존성 설치:

```bash
pip install -r requirements.txt
```

3. `.env` 파일 설정 (웹훅 URL 등)

### 로컬 실행

```bash
python app.py
```

서버가 http://localhost:3001 에서 실행됩니다.

### Docker로 실행하기 (추천)

#### 1. Docker 설치
- **Mac**: [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop) 다운로드 및 설치
- **Windows**: [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop) 다운로드 및 설치
- **Linux**: 공식 Docker 설치 가이드 참조

#### 2. Docker로 로컬 실행
```bash
# Docker 이미지 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 중지
docker-compose down
```

서버가 http://localhost:3001 에서 실행됩니다.

#### 3. Docker 프로덕션 배포

**Google Cloud Run 배포:**
```bash
# Google Cloud SDK 설치 후
gcloud builds submit --tag gcr.io/PROJECT-ID/face-analysis
gcloud run deploy face-analysis --image gcr.io/PROJECT-ID/face-analysis --platform managed --region asia-northeast3 --allow-unauthenticated
```

**Render.com 배포:**
1. https://render.com 에 로그인
2. "New Web Service" 선택
3. GitHub 레포지토리 연결
4. "Docker" 선택
5. 환경변수 설정 후 배포

**Fly.io 배포:**
```bash
# Fly CLI 설치 후
fly launch
fly deploy
```

### API
- POST `/api/face-analysis` with form-data `image` (file): returns facial feature analysis and a visualization image (background blur, face focus).

Example using curl:

```bash
curl -X POST http://localhost:3000/api/face-analysis \
  -F "image=@/path/to/your/face.jpg"
```

Response body:

```json
{
  "status": "success",
  "message": "얼굴 분석이 완료되었습니다.",
  "basic_analysis": {
    "eyebrows": "...",
    "eyes": "...",
    "nose": "...",
    "mouth": "...",
    "jaw": "..."
  },
  "vis_image": "data:image/jpeg;base64,...",
  "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```

### Notes
- Uses MediaPipe Face Mesh with refined landmarks.
- Accepts PNG/JPG/JPEG/GIF/BMP.


