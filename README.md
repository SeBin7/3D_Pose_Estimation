# 📊 자세 교정 보조앱 – Posture Corrector (with OpenVINO)

🧘‍♀️ 앉은 자세에서 "허리 구부정", "거북목", "라운드 숄더"와 같은 나쁜 자세를 실시간으로 감지하여
사용자에게 LED 또는 메시지로 경고하는 자세 교정 보조 시스템입니다.

본 프로젝트는 Intel OpenVINO의 3D Pose Estimation 예제를 기반으로,
실생활에 유용한 AI 활용 사례를 구현한 미니 프로젝트입니다.

---

## ✅ 프로젝트 목적

- OpenVINO 예제를 실제 생활 문제에 적용해 보기
- 3D Pose Estimation을 통해 앉은 자세의 주요 문제 감지
- 지속된 나쁜 자세에 대해 경고해 바른 자세 유도

---

## 🧠 주요 기능

| 기능 | 설명 |
|------|------|
| 3D 자세 추정 | OpenVINO의 `human-pose-estimation-3d` 모델을 통해 keypoint 추출 |
| EMA 기반 판단 | 튀는 값을 보정하기 위해 지수 이동 평균(EMA) 적용 |
| 나쁜 자세 탐지 | 허리 각도, 머리/엉덩이의 Z좌표 기반으로 판단 |
| 10초 지속 감지 | 나쁜 자세가 10초 이상 지속되면 경고 출력 |
| 모듈화된 구조 | Pose 추론 / 자세 분석 / 시각화 모듈로 분리되어 관리 용이 |

---

## 🏗️ 모듈 구조

```
PostureCorrector/
├── posture_corrector/
│   ├── __init__.py
│   ├── pose_model.py          # OpenVINO 모델 로딩 및 추론
│   ├── posture_analyzer.py   # 자세 판단 + EMA + 타이머 관리
│   └── visualizer.py         # 화면 출력 및 FPS 표시
│
├── main.py                   # 실행 스크립트
├── README.md
├── requirements.txt
```

---

## 📏 판단 기준

| 자세 유형 | 판단 지표 | 기준값 (EMA) |
|-----------|-----------|----------------|
| 허리 구부정 | 목–엉덩이 각도 | > 35도 |
| 거북목 | 목과 코의 z축 차이 | > 0.03 |
| 라운드 숄더 | 어깨–엉덩이 z축 차이 | > 0.02 |

→ 모든 기준은 **EMA 필터링**을 거친 후, **10초 이상 지속 시** 경고 출력합니다.

---

## ▶️ 실행 방법

1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. OpenVINO 환경 활성화
```bash
source openvino_env/bin/activate  # 가상환경명에 따른 변경
```

3. 실행
```bash
python main.py
```



## 💡 향후 확장 방향

- `cv2.putText()`로 시각적 메시지 제공
- LED 경고 출력 or LLM 자세 조언 출력
- 자세 교정 로그 저장 기능


