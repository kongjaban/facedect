from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from datetime import datetime
import os
import cv2
import numpy as np
import base64
from PIL import Image
import json
import requests
import mediapipe as mp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-default-secret-key')

# MediaPipe 초기화 객체
mp_face_mesh = mp.solutions.face_mesh


def _is_face_quality_ok(face_landmarks, image_shape):
    """얼굴 품질 검사 (완화된 기준)"""
    h, w = image_shape[:2]

    if not face_landmarks:
        return False, "얼굴을 찾을 수 없습니다."

    x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in face_landmarks.landmark]

    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)

    if face_width < 50 or face_height < 50:
        return False, "얼굴이 너무 작습니다. 더 가까이에서 촬영해주세요."

    margin = 5
    if (
        min(x_coords) < margin
        or min(y_coords) < margin
        or max(x_coords) > w - margin
        or max(y_coords) > h - margin
    ):
        # 경고만 출력하고 계속 진행
        print("⚠️ 얼굴이 이미지 경계에 가깝습니다. 분석을 계속합니다.")

    return True, "OK"


def create_face_focus_image(image, face_landmarks):
    """얼굴 영역만 선명하게 하고 배경은 흐리게 처리한 이미지 생성"""
    h, w = image.shape[:2]

    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]

    mask = np.zeros((h, w), dtype=np.uint8)

    face_points = []
    for idx in FACE_OVAL:
        if idx < len(face_landmarks.landmark):
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            face_points.append([x, y])

    if len(face_points) > 3:
        face_points = np.array(face_points)
        center_x = np.mean(face_points[:, 0])
        center_y = np.mean(face_points[:, 1])

        expanded_points = []
        for point in face_points:
            dx = point[0] - center_x
            dy = point[1] - center_y
            new_x = center_x + dx * 1.3
            new_y = center_y + dy * 1.3
            expanded_points.append([int(new_x), int(new_y)])

        expanded_points = np.array(expanded_points)
        cv2.fillPoly(mask, [expanded_points], 255)

    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_normalized = mask_blur.astype(np.float32) / 255.0

    background_blur = cv2.GaussianBlur(image, (51, 51), 0).astype(np.float32)
    result = image.copy().astype(np.float32)

    for c in range(3):
        result[:, :, c] = result[:, :, c] * mask_normalized + background_blur[:, :, c] * (1 - mask_normalized)

    result = result.astype(np.uint8)

    if len(face_points) > 3:
        cv2.polylines(result, [expanded_points], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(result, [expanded_points], True, (100, 149, 237), 1, cv2.LINE_AA)

    # Draw all MediaPipe face landmarks
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Draw small circles for each landmark point
        cv2.circle(result, (x, y), 1, (0, 255, 0), -1)  # Green dots
        cv2.circle(result, (x, y), 2, (0, 0, 0), 1)     # Black outline

    return result


def analyze_facial_features(landmarks, image_shape):
    h, w = image_shape[:2]

    def get_distance(p1_idx, p2_idx):
        if p1_idx >= len(landmarks) or p2_idx >= len(landmarks):
            return 0
        p1 = landmarks[p1_idx]
        p2 = landmarks[p2_idx]
        return float(np.linalg.norm(p1 - p2))

    def get_width_height(indices):
        valid_indices = [i for i in indices if i < len(landmarks)]
        if not valid_indices:
            return 0, 0
        points = landmarks[valid_indices]
        width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        return width, height

    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYEBROW = [296, 334, 293, 300, 276, 285, 295, 282, 283, 276]
    LEFT_EYE = [33, 133, 159, 145]
    RIGHT_EYE = [362, 263, 386, 374]
    LEFT_EYE_TAIL = [33, 133]
    RIGHT_EYE_TAIL = [362, 263]
    MOUTH_OUTER = [61, 291, 13, 14, 78, 308, 87, 317]
    CHIN = [18, 175, 199]

    face_width = get_distance(234, 454)
    face_height = get_distance(10, 152)

    left_brow_width, left_brow_height = get_width_height(LEFT_EYEBROW)
    right_brow_width, right_brow_height = get_width_height(RIGHT_EYEBROW)
    avg_brow_thickness = (left_brow_height + right_brow_height) / 2 if (left_brow_height + right_brow_height) else 0
    brow_thickness_ratio = (avg_brow_thickness / face_height) if face_height > 0 else 0
    left_brow_angle = landmarks[LEFT_EYEBROW[-1]][1] - landmarks[LEFT_EYEBROW[0]][1]
    right_brow_angle = landmarks[RIGHT_EYEBROW[-1]][1] - landmarks[RIGHT_EYEBROW[0]][1]
    brow_slope = (left_brow_angle + right_brow_angle) / 2
    if brow_thickness_ratio > 0.015:
        eyebrow_density = "굵은 눈썹"
    elif brow_thickness_ratio < 0.008:
        eyebrow_density = "얇은 눈썹"
    else:
        eyebrow_density = "보통 눈썹"
    if brow_slope > 3:
        eyebrow_shape = "아치형 눈썹"
    elif brow_slope < -3:
        eyebrow_shape = "일자형 눈썹"
    else:
        eyebrow_shape = "살짝 곡선형 눈썹"
    eyebrow_type = f"{eyebrow_density}, {eyebrow_shape}"

    left_eye_width = get_distance(LEFT_EYE[0], LEFT_EYE[1])
    right_eye_width = get_distance(RIGHT_EYE[0], RIGHT_EYE[1])
    left_eye_height = get_distance(LEFT_EYE[2], LEFT_EYE[3])
    right_eye_height = get_distance(RIGHT_EYE[2], RIGHT_EYE[3])
    avg_eye_width = (left_eye_width + right_eye_width) / 2 if (left_eye_width + right_eye_width) else 0
    avg_eye_height = (left_eye_height + right_eye_height) / 2 if (left_eye_height + right_eye_height) else 0
    eye_aspect_ratio = (avg_eye_width / avg_eye_height) if avg_eye_height > 0 else 0
    eye_ratio = (avg_eye_width / face_width) if face_width > 0 else 0
    left_eye_angle = landmarks[LEFT_EYE_TAIL[0]][1] - landmarks[LEFT_EYE_TAIL[1]][1]
    right_eye_angle = landmarks[RIGHT_EYE_TAIL[0]][1] - landmarks[RIGHT_EYE_TAIL[1]][1]
    if left_eye_angle < -2 and right_eye_angle < -2:
        eye_tail = "눈꼬리 올라감"
    elif left_eye_angle > 2 and right_eye_angle > 2:
        eye_tail = "눈꼬리 내려감"
    else:
        eye_tail = "평평한 눈꼬리"
    eyelid_opening_ratio = (avg_eye_height / face_height) if face_height > 0 else 0
    if eyelid_opening_ratio > 0.05:
        eyelid_type = "쌍꺼풀 있음"
    elif eyelid_opening_ratio < 0.03:
        eyelid_type = "무쌍꺼풀"
    else:
        eyelid_type = "얇은 쌍꺼풀"
    if eye_ratio > 0.08:
        if eye_aspect_ratio > 3:
            eye_size_type = "가는 눈"
        else:
            eye_size_type = "큰 눈"
    elif eye_ratio < 0.06:
        eye_size_type = "작은 눈"
    else:
        eye_size_type = "보통 눈"
    eye_type = f"{eye_size_type}, {eye_tail}, {eyelid_type}"

    nose_length = get_distance(1, 6)
    nose_width = get_distance(31, 35)
    nose_bridge_height = get_distance(168, 195)
    nose_ratio = (nose_width / face_width) if face_width > 0 else 0
    length_ratio = (nose_length / face_height) if face_height > 0 else 0
    bridge_ratio = (nose_bridge_height / face_height) if face_height > 0 else 0
    if length_ratio > 0.16:
        nose_length_type = "긴 코"
    elif length_ratio < 0.12:
        nose_length_type = "짧은 코"
    else:
        nose_length_type = "보통 길이 코"
    if nose_ratio > 0.15:
        nose_width_type = "넓은 코"
    elif nose_ratio < 0.10:
        nose_width_type = "좁은 코"
    else:
        nose_width_type = "보통 너비 코"
    if bridge_ratio > 0.05:
        nose_bridge_type = "높은 콧대"
    elif bridge_ratio < 0.03:
        nose_bridge_type = "낮은 콧대"
    else:
        nose_bridge_type = "보통 콧대"
    nose_type = f"{nose_length_type}, {nose_width_type}, {nose_bridge_type}"

    mouth_width = get_distance(61, 291)
    mouth_opening = get_distance(13, 14)
    upper_lip_thickness = get_distance(78, 13)
    lower_lip_thickness = get_distance(14, 308)
    total_lip_thickness = upper_lip_thickness + lower_lip_thickness
    mouth_ratio = (mouth_width / face_width) if face_width > 0 else 0
    mouth_open_ratio = (mouth_opening / face_height) if face_height > 0 else 0
    lip_thickness_ratio = (total_lip_thickness / face_height) if face_height > 0 else 0
    if mouth_ratio > 0.12:
        mouth_size = "큰 입"
    elif mouth_ratio < 0.09:
        mouth_size = "작은 입"
    else:
        mouth_size = "보통 입"
    if lip_thickness_ratio > 0.06:
        lip_type = "두꺼운 입술"
    elif lip_thickness_ratio < 0.03:
        lip_type = "얇은 입술"
    else:
        lip_type = "보통 입술"
    if mouth_open_ratio > 0.05:
        mouth_expression = "입 벌어짐 있음"
    else:
        mouth_expression = "닫힌 입"
    mouth_type = f"{mouth_size}, {lip_type}, {mouth_expression}"

    jaw_width = get_distance(172, 397)
    chin_width, chin_height = get_width_height(CHIN)
    chin_ratio = (chin_height / face_height) if face_height > 0 else 0
    jaw_ratio = (jaw_width / face_width) if face_width > 0 else 0
    chin_shape_ratio = (chin_width / chin_height) if chin_height > 0 else 0
    if chin_ratio > 0.15:
        chin_proj = "돌출된 턱"
    elif chin_ratio < 0.08:
        chin_proj = "들어간 턱"
    else:
        chin_proj = "보통 돌출 턱"
    if jaw_ratio > 0.55:
        jaw_width_type = "각진 턱"
    elif jaw_ratio < 0.45:
        jaw_width_type = "갸름한 턱"
    else:
        jaw_width_type = "보통 턱"
    if chin_shape_ratio > 1.2:
        chin_shape_type = "긴 턱"
    elif chin_shape_ratio < 0.8:
        chin_shape_type = "짧은 턱"
    else:
        chin_shape_type = "중간 길이 턱"
    jaw_type = f"{chin_proj}, {jaw_width_type}, {chin_shape_type}"

    metrics = {
        "face": {
            "face_width": face_width,
            "face_height": face_height,
        },
        "eyebrows": {
            "left_height": left_brow_height,
            "right_height": right_brow_height,
            "avg_thickness": avg_brow_thickness,
            "thickness_ratio": brow_thickness_ratio,
            "left_slope": left_brow_angle,
            "right_slope": right_brow_angle,
            "avg_slope": brow_slope,
        },
        "eyes": {
            "left_width": left_eye_width,
            "right_width": right_eye_width,
            "left_height": left_eye_height,
            "right_height": right_eye_height,
            "avg_width": avg_eye_width,
            "avg_height": avg_eye_height,
            "aspect_ratio": eye_aspect_ratio,
            "width_to_face_ratio": eye_ratio,
            "left_tail_angle": left_eye_angle,
            "right_tail_angle": right_eye_angle,
            "eyelid_opening_ratio": eyelid_opening_ratio,
        },
        "nose": {
            "length": nose_length,
            "width": nose_width,
            "bridge_height": nose_bridge_height,
            "width_to_face_ratio": nose_ratio,
            "length_to_face_ratio": length_ratio,
            "bridge_to_face_ratio": bridge_ratio,
        },
        "mouth": {
            "width": mouth_width,
            "opening": mouth_opening,
            "upper_lip": upper_lip_thickness,
            "lower_lip": lower_lip_thickness,
            "total_lip": total_lip_thickness,
            "width_to_face_ratio": mouth_ratio,
            "open_to_face_ratio": mouth_open_ratio,
            "lip_to_face_ratio": lip_thickness_ratio,
        },
        "jaw": {
            "jaw_width": jaw_width,
            "chin_width": chin_width,
            "chin_height": chin_height,
            "chin_to_face_ratio": chin_ratio,
            "jaw_to_face_ratio": jaw_ratio,
            "chin_shape_ratio": chin_shape_ratio,
        },
    }

    return {
        "eyebrows": eyebrow_type,
        "eyes": eye_type,
        "nose": nose_type,
        "mouth": mouth_type,
        "jaw": jaw_type,
        "metrics": metrics,
    }


def analyze_face(image):
    """MediaPipe Face Mesh 기반 얼굴 분석"""
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return None, "얼굴을 찾을 수 없습니다. 다시 촬영해 주세요."

            face_landmarks = results.multi_face_landmarks[0]

            ok, msg = _is_face_quality_ok(face_landmarks, image.shape)
            if not ok:
                return None, msg

            h, w = image.shape[:2]
            landmarks = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append([x, y])
            landmarks = np.array(landmarks)

            analysis_result = analyze_facial_features(landmarks, image.shape)
            vis_image = create_face_focus_image(image, face_landmarks)

            _, buf = cv2.imencode('.jpg', vis_image)
            img_b64 = base64.b64encode(buf).decode('utf-8')

            # MediaPipe face landmarks coordinates (normalized 0-1)
            landmarks_coords = []
            for lm in face_landmarks.landmark:
                landmarks_coords.append({
                    "x": float(lm.x),
                    "y": float(lm.y), 
                    "z": float(lm.z)
                })

            return {
                "eyebrows": analysis_result["eyebrows"],
                "eyes": analysis_result["eyes"],
                "nose": analysis_result["nose"],
                "mouth": analysis_result["mouth"],
                "jaw": analysis_result["jaw"],
                "metrics": analysis_result.get("metrics", {}),
                "landmarks": landmarks_coords,
                "vis_image": f"data:image/jpeg;base64,{img_b64}",
            }, None
    except Exception as e:
        print(f"얼굴 분석 오류: {e}")
        return None, f"얼굴 분석 중 오류가 발생했습니다: {str(e)}"


def _to_python_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.route('/')
def index():
    # 비회원(게스트)도 메인 페이지 접근 가능
    return render_template('index.html')


@app.route('/login')
def login():
    if 'user' in session and session['user'].get('is_logged_in'):
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        webhook_url = os.environ.get('SIGNUP_WEBHOOK_URL', 'https://sijinn8n.app.n8n.cloud/webhook/cd49d1ea-4700-48e4-8df4-40983abaa991')
        # 외부 서비스 요청 타임아웃 60초
        response = requests.post(webhook_url, json=data, timeout=60)
        if response.status_code == 200:
            session['user'] = {**(data or {}), 'is_logged_in': True}
            return jsonify({'status': 'success', 'message': '회원가입이 완료되었습니다.'})
        return jsonify({'status': 'error', 'message': '회원가입 중 오류가 발생했습니다.'}), 500
    except Exception as e:
        print(f"Signup Error: {e}")
        return jsonify({'status': 'error', 'message': '서버 오류가 발생했습니다.'}), 500


@app.route('/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        webhook_url = os.environ.get('SIGNIN_WEBHOOK_URL', 'https://sijinn8n.app.n8n.cloud/webhook/2f11a0b8-4a2b-417f-a7a2-efd5b8d28614')
        # 외부 서비스 요청 타임아웃 60초
        response = requests.post(webhook_url, json=data, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('status') == 'success':
                session['user'] = {**response_data.get('user', {}), 'email': data.get('email'), 'is_logged_in': True}
                return jsonify({'status': 'success', 'message': '로그인되었습니다.'})
            return jsonify({'status': 'error', 'message': response_data.get('message', '이메일 또는 비밀번호가 올바르지 않습니다.')}), 401
        return jsonify({'status': 'error', 'message': '로그인 서비스에 문제가 발생했습니다.'}), 500
    except Exception as e:
        print(f"Signin Error: {e}")
        return jsonify({'status': 'error', 'message': '서버 오류가 발생했습니다.'}), 500


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/api/user')
def get_user_info():
    if 'user' in session and session['user'].get('is_logged_in'):
        return jsonify({'status': 'success', 'user': session['user']})
    # 게스트 사용자 정보 반환
    guest_user = {
        'name': '게스트',
        'email': '',
        'is_logged_in': False
    }
    return jsonify({'status': 'success', 'user': guest_user})


def get_face_reading_from_n8n(analysis_result, user_info=None):
    try:
        webhook_data = {
            "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "facial_features": {
                "eyebrows": analysis_result.get("eyebrows", ""),
                "eyes": analysis_result.get("eyes", ""),
                "nose": analysis_result.get("nose", ""),
                "mouth": analysis_result.get("mouth", ""),
                "jaw": analysis_result.get("jaw", "")
            }
        }
        # 사용자 정보(옵션)를 포함 (user_id 유무와 관계없이)
        if user_info:
            webhook_data["user_info"] = user_info

        webhook_url = os.environ.get(
            'N8N_FACE_READING_URL',
            "https://sijinn8n.app.n8n.cloud/webhook/db346cbb-5e5a-4afa-84f5-4485ff8b4ff3",
        )
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        print(f"[n8n] POST {webhook_url}")
        print(f"[n8n] Payload: {json.dumps(webhook_data, ensure_ascii=False)}")
        # n8n 관상 해석 요청 타임아웃 60초
        response = requests.post(webhook_url, json=webhook_data, headers=headers, timeout=60)
        print(f"[n8n] Status: {response.status_code}")
        try:
            print(f"[n8n] Body: {response.text}")
        except Exception:
            pass

        if response.status_code == 200:
            response_data = response.json()
            if isinstance(response_data, dict):
                if 'output' in response_data:
                    output_str = response_data['output']
                    # 1) Markdown 코드펜스 안의 JSON
                    if isinstance(output_str, str) and '```json' in output_str:
                        start_idx = output_str.find('```json') + 7
                        end_idx = output_str.find('```', start_idx)
                        if end_idx != -1:
                            json_str = output_str[start_idx:end_idx].strip()
                            return {"status": "success", "interpretation": json.loads(json_str)}
                    # 2) 코드펜스 없이 순수 JSON 문자열인 경우 파싱 시도
                    if isinstance(output_str, str) and output_str.strip().startswith('{'):
                        try:
                            return {"status": "success", "interpretation": json.loads(output_str)}
                        except json.JSONDecodeError:
                            pass
                    # 3) 그 외에는 전체 텍스트를 종합 해석으로 간주
                    return {"status": "success", "interpretation": {"overall_reading": output_str}}
                if 'message' in response_data and 'content' in response_data['message']:
                    content_str = response_data['message']['content']
                    if content_str.strip().startswith('```json'):
                        content_str = content_str.strip().replace('```json', '', 1)
                        content_str = content_str.rsplit('```', 1)[0].strip()
                    try:
                        return {"status": "success", "interpretation": json.loads(content_str)}
                    except json.JSONDecodeError:
                        return {"status": "success", "interpretation": {"overall_reading": content_str}}
                if 'overall_reading' in response_data:
                    return {"status": "success", "interpretation": response_data}
            return {"status": "success", "interpretation": response_data}
        return {"status": "error", "message": f"n8n 오류 (HTTP {response.status_code})"}
    except requests.exceptions.RequestException as e:
        print(f"관상 해석 요청 실패: {e}")
        return {"status": "error", "message": f"n8n 연결 실패: {str(e)}"}
    except Exception as e:
        print(f"관상 해석 오류: {e}")
        return {"status": "error", "message": f"n8n 처리 오류: {str(e)}"}


# 기본(가짜) 해석은 미사용. 실데이터(n8n)만 사용.


@app.route('/api/face-analysis', methods=['POST'])
def face_analysis():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "이미지 파일이 없습니다."}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "파일이 선택되지 않았습니다."}), 400

        allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed):
            return jsonify({"status": "error", "message": "지원하지 않는 파일 형식입니다."}), 400

        try:
            pil_image = Image.open(file.stream)
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            image_array = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception:
            return jsonify({"status": "error", "message": "이미지 파일을 처리할 수 없습니다."}), 400

        result, error_msg = analyze_face(image_bgr)
        if result is None:
            return jsonify({"status": "error", "message": error_msg}), 400

        # 세션 기반 사용자 정보
        user_info = session.get('user', {}) if ('user' in session and session['user'].get('is_logged_in')) else {}

        # 폼에서 전달된 옵션 정보 병합 (비회원도 사용 가능)
        form_gender = request.form.get('gender')
        form_birth_date = request.form.get('birth_date')
        form_birth_time = request.form.get('birth_time')
        # 기본 구조 보장
        if not isinstance(user_info, dict):
            user_info = {}
        if form_gender:
            user_info['gender'] = form_gender
        if form_birth_date:
            user_info['birth_date'] = form_birth_date
        if form_birth_time:
            user_info['birth_time'] = form_birth_time
        face_reading_result = get_face_reading_from_n8n(result, user_info)

        response_data = {
            'status': 'success',
            'message': '분석이 완료되었습니다.',
            'basic_analysis': {
                'eyebrows': result.get('eyebrows', ''),
                'eyes': result.get('eyes', ''),
                'nose': result.get('nose', ''),
                'mouth': result.get('mouth', ''),
                'jaw': result.get('jaw', ''),
            },
            'metrics': result.get('metrics', {}),
            'vis_image': result.get('vis_image', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # n8n 응답만 사용 (실데이터). 실패 시 에러로 반환
        if face_reading_result.get('status') == 'success':
            response_data['face_reading'] = face_reading_result.get('interpretation')
        else:
            return jsonify({"status": "error", "message": "관상 해석 서비스 오류: 리얼 응답을 받지 못했습니다."}), 502

        # Ensure JSON serializable types
        response_data = _to_python_types(response_data)
        return jsonify(response_data)
    except Exception as e:
        print(f"💥 얼굴 분석 API 오류: {e}")
        return jsonify({"status": "error", "message": "얼굴 분석 중 오류가 발생했습니다."}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)


