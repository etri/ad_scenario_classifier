# 시나리오 분류기 (Scenario Classifier)

이 프로젝트는 NuPlan 프레임워크 및 실제 자율주행 로그 데이터를 위한 자동화된 주행 시나리오 분류 및 라벨링 파이프라인입니다. JSON 형식 로그를 입력받아 다양한 주행 상황(차선변경, 정지, 회전 등)을 규칙 기반 다단계 파이프라인으로 분류하고, 시각화 및 결과 데이터를 제공합니다.

## 📋 개요

- 자율주행 로그 데이터의 다양한 주행 시나리오를 자동으로 식별 및 분류
- 대용량 로그를 확장성 있게 처리하고 다양한 시나리오로의 확장 용이
- NuPlan 데이터 구조를 기반으로 SQLite 맵 정보를 활용한 정확한 공간 분석 지원
- 시나리오 라벨링 결과와 데이터를 시각화 이미지로 출력
- Rule-based 4단계 분류 (상태, 행동, 상호작용, 동역학 기반)

## 주요 기능

- **자동 시나리오 분류**: 101-epoch (과거 40 + 현재 1 + 미래 60) 윈도우 기반 주행 구간 단위 분류
- **맵 기반 정밀 분석**: 차선, 정지선, 교차로, 신호 등 공간 요소 자동 인식
- **신뢰도 기반 라벨링**: 각 라벨별 신뢰도(0.0~1.0) 산출
- **Ego 중심 시각화**: 각 시나리오 별 60x60m 이내 객체/궤적/라벨 이미지 생성

## 설치 및 요구사항

- Python 3.8 이상 필요
- 추가 파이썬 패키지 및 의존성은 `requirements.txt` 참고

```bash
git clone https://github.com/yourusername/ad_scenario_classifier.git
cd ad_scenario_classifier
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate
pip install -r requirements.txt
```

## 사용 예시

```bash
python main.py --input <로그_파일_경로> --output <결과_저장_경로>
# 옵션 예시:
python main.py --input data/logs/ --output results/ --model <모델_경로>
```

## 프로젝트 구조

```
ad_scenario_classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── data/
├── results/
└── tests/
```

## 📚 기술 문서 안내

자세한 구조 및 구현 설명은 `doc/` 디렉토리 참고

1. **[01_architecture.md](doc/01_architecture.md)**: 전체 시스템 데이터 흐름과 컴포넌트(예: JsonLogLoader, MapManager, ScenarioWindow Extraction 등) 구조
2. **[02_data_structures.md](doc/02_data_structures.md)**: ScenarioLabel, ScenarioWindow 등 핵심 데이터 타입 정의
3. **[03_scenario_labeler.md](doc/03_scenario_labeler.md)**: 4단계 Rule-based 분류 알고리즘 설명
4. **[04_map_manager.md](doc/04_map_manager.md)**: SQLite 기반 맵 데이터 접근, STRtree 공간 인덱싱 및 쿼리
5. **[05_json_loader.md](doc/05_json_loader.md)**: 로그 파일 NuPlan 구조 파싱 및 변환법
6. **[06_visualization.md](doc/06_visualization.md)**: Ego 좌표계 시각화, 궤적/레이어 렌더링 방법
7. **[07_export.md](doc/07_export.md)**: 라벨링 결과 JSON 구조와 출력 방식
8. **[08_pipeline.md](doc/08_pipeline.md)**: 전체 처리 흐름 및 최적화 제안

### 분류 라벨 예시

- 속도: `low_magnitude_speed`, `medium_magnitude_speed`, `high_magnitude_speed`
- 상태: `stationary`, `stationary_in_traffic`
- 회전: `starting_left_turn`, `starting_right_turn`, `starting_high_speed_turn`, `starting_low_speed_turn`
- 차선변경: `changing_lane`, `changing_lane_to_left`, `changing_lane_to_right`
- 추종: `following_lane_with_lead`, `following_lane_with_slow_lead`, `following_lane_without_lead`
- 근접/상호작용: `near_high_speed_vehicle`, `near_long_vehicle`, `near_multiple_vehicles`, `near_construction_zone_sign`, `near_trafficcone_on_driveable`, `near_barrier_on_driveable`, `behind_long_vehicle`, `behind_bike`, `near_multiple_pedestrians`
- 동역학: `high_magnitude_jerk`, `high_lateral_acceleration`
- 맵 기반: `on_stopline_traffic_light`, `on_stopline_stop_sign`, `on_stopline_crosswalk`, `on_intersection`, `on_traffic_light_intersection`, `on_all_way_stop_intersection`

## 시스템 아키텍처 요약

- **입력**: JSON 로그 파일
- **처리**: Log Loader → MapManager → ScenarioWindow Extraction → ScenarioLabeler(4단계 분류) → ScenarioExporter/Visualizer
- **출력**: 라벨링된 JSON, 시각화 PNG

## 데이터 구조

- **ScenarioWindow**: 101-epoch 단위 시나리오 묶음
- **ScenarioLabel**: 라벨/신뢰도/카테고리 묶음
- **LabeledScenario**: 결과 JSON용 시나리오
- **LogEntry**: 단일 타임스탬프의 원시 정보

## 처리 파이프라인

1. 로그 파일 로딩 및 파싱
2. 맵 데이터베이스 초기화
3. 101-epoch 시나리오 윈도우 생성
4. Rule-based 4단계 분류 실행
5. 라벨링 결과 JSON 저장
6. 시각화 이미지 생성

## 설정

주요 파라미터는 `src/DefaultParams.py`에서 관리됩니다.
- 로그 파일/맵 파일 경로, 출력 디렉토리, 처리 옵션 등

## 참고 자료

- [NuPlan 프레임워크](https://github.com/motional/nuplan-devkit)
- [Shapely](https://shapely.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일 참고.

## 🤝 기여 방법

1. 저장소를 포크합니다
2. 새 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📧 문의

이슈를 통해 질문이나 제안사항을 남겨주세요.

---

**참고**: 프로젝트와 문서는 계속 업데이트될 예정입니다.
