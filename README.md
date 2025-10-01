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

### Railway 배포 방법

1. **Railway 프로젝트 생성**
   - https://railway.app 에 로그인
   - "New Project" → "Deploy from GitHub repo" 선택

2. **환경변수 설정**
   Railway 대시보드에서 다음 환경변수를 설정하세요:
   ```
   SIGNUP_WEBHOOK_URL=https://sijinn8n.app.n8n.cloud/webhook/cd49d1ea-4700-48e4-8df4-40983abaa991
   SIGNIN_WEBHOOK_URL=https://sijinn8n.app.n8n.cloud/webhook/2f11a0b8-4a2b-417f-a7a2-efd5b8d28614
   N8N_FACE_READING_URL=https://sijinn8n.app.n8n.cloud/webhook/db346cbb-5e5a-4afa-84f5-4485ff8b4ff3
   SECRET_KEY=랜덤-비밀키-생성하세요
   FLASK_ENV=production
   PORT=3001
   ```

3. **자동 배포**
   - GitHub에 push하면 Railway가 자동으로 배포합니다
   - Procfile과 railway.json이 설정을 자동으로 처리합니다

4. **도메인 설정**
   - Railway 대시보드에서 자동 생성된 도메인을 사용하거나
   - 커스텀 도메인을 연결할 수 있습니다

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


