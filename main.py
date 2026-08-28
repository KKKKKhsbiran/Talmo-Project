from fastapi import FastAPI, UploadFile, Form, Header, File, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import hashlib
import os

app = FastAPI()

# 마스터 플랜: 토큰은 환경변수에서만 읽는다 (하드코딩 금지)
# Render 환경변수에서 토큰을 가져오고, 로컬 테스트용 기본값 지정
API_TOKEN = os.getenv("MODEL_API_TOKEN", "default_secret_token")
# 배포 플랫폼(Replicate/HF Spaces 등)의 환경변수 설정에 MODEL_API_TOKEN을 등록해야 함
EXPECTED_TOKEN = f"Bearer {API_TOKEN}"

if not EXPECTED_TOKEN:
    # 서버 기동 시점에 바로 경고 — 배포 환경에서 토큰 설정을 깜빡하는 실수를 조기에 발견하기 위함
    print("[WARNING] MODEL_API_TOKEN 환경변수가 설정되지 않았습니다. 인가 검사가 항상 실패합니다.")


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    view: str = Form(...),
    request_id: str = Form(...),
    authorization: str = Header(None),
    force: str = Query(None)  # 테스트용 플래그: ?force=quality_fail 또는 ?force=error
):
    # 1. 보안 인가 (인가되지 않은 직접 호출 차단)
    expected_header = f"Bearer {EXPECTED_TOKEN}" if EXPECTED_TOKEN else None
    if not expected_header or authorization != expected_header:
        return JSONResponse(
            status_code=401,
            content={"ok": False, "error": {"code": "UNAUTHORIZED", "message": "토큰이 없습니다."}}
        )

    # 2. 4xx/5xx 강제 에러 발생 테스트 (?force=error)
    if force == "error":
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": {"code": "IMAGE_TOO_LARGE", "message": "Simulated error: Image exceeds 8MB."}}
        )

    # 3. 화질 불량 테스트 (?force=quality_fail)
    if force == "quality_fail":
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "quality": {"pass": False, "issues": ["too_dark", "blurry"]},
                "stage": None,
                "indices": {},
                "progress_index": None,
                "confidence": None,
                "model_version": "v0.1.0-mock"
            }
        )

    # 4. 결정성 보장: request_id 기반 해싱 (같은 사진=같은 값)
    hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # 5. 모델 출력 (노우드 1~7 기반)
    mock_stage = (hash_val % 7) + 1  # 내부용 정수 등급 (1~7)

    # stage와 progress_index가 서로 다른 해시 구간에서 독립적으로 나오지 않도록,
    # progress_index를 stage 기준 구간 안에서 산출하도록 변경 (예: stage=3 -> 28.6~42.9 사이)
    stage_span = 100.0 / 7
    stage_lower = (mock_stage - 1) * stage_span
    mock_progress = round(stage_lower + (hash_val % 1000) / 1000.0 * stage_span, 1)

    mock_confidence = round(0.70 + (hash_val % 30) / 100.0, 2)

    # 6. 정상 200 응답 (의학 용어 완벽 배제)
    return {
        "ok": True,
        "request_id": request_id,
        "views_received": [view],       # 향후 정면(frontal) 확장 시 프론트 수정 불필요
        "completeness": 1.0,
        "quality": {"pass": True, "issues": []},
        "stage": mock_stage,            # 프론트는 이 값을 문장("노우드 3단계")으로 변환 금지
        "indices": {view: mock_progress},
        "progress_index": mock_progress,
        "confidence": mock_confidence,
        "model_version": "v0.1.0-mock",
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")   
    }
