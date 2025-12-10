# 시나리오 분류기 (Scenario Classifier) - 상세 문서

이 문서는 시나리오 분류기 프로젝트의 상세 기술 문서입니다. 각 모듈과 컴포넌트에 대한 심층적인 설명을 제공합니다.

## 문서 목차

1. [시스템 아키텍처](./01_architecture.md) - 전체 시스템 구조 및 데이터 흐름
2. [데이터 구조](./02_data_structures.md) - 핵심 데이터 타입 상세 설명
3. [시나리오 분류기](./03_scenario_labeler.md) - 4단계 분류 알고리즘 상세
4. [맵 관리자](./04_map_manager.md) - SQLite 기반 맵 데이터 관리
5. [JSON 로그 로더](./05_json_loader.md) - 로그 파일 로딩 및 변환
6. [시각화 시스템](./06_visualization.md) - 시나리오 시각화 엔진
7. [데이터 출력](./07_export.md) - JSON 출력 형식 및 구조
8. [처리 파이프라인](./08_pipeline.md) - 전체 처리 흐름 상세

## 프로젝트 개요

시나리오 분류기는 NuPlan 프레임워크를 기반으로 자율주행 로그 데이터에서 주행 시나리오를 자동으로 분류하고 라벨링하는 시스템입니다. JSON 형식의 주행 로그를 입력받아 다양한 주행 상황(차선변경, 정지, 회전 등)을 식별하고 시각화합니다.

### 핵심 특징

- **Rule-based 분류**: 규칙 기반 4단계 분류 파이프라인
- **101-epoch 윈도우**: 과거 40프레임 + 현재 1프레임 + 미래 60프레임
- **맵 기반 분석**: SQLite 맵 데이터를 활용한 정확한 위치 분석
- **신뢰도 기반 라벨링**: 각 라벨에 0.0~1.0 범위의 신뢰도 할당
- **시각화**: Ego 중심 좌표계 기반 시나리오 시각화

## 주요 기능

### 시나리오 분류

4단계 분류 파이프라인을 통해 다양한 주행 시나리오를 자동으로 식별합니다:

1. **명시적 상태 분류** (State-based)
   - 속도 프로파일 (저속/중속/고속)
   - 정지 상태 감지
   - 맵 기반 위치 분류 (정지선, 교차로 등)

2. **궤적 기반 행동 분류** (Behavior-based)
   - 차선변경 감지
   - 회전 감지 (좌회전/우회전, 고속/저속)

3. **상호작용 기반 분류** (Interaction-based)
   - 선행차 추종
   - 다중 차량/보행자 근접
   - 특수 차량 인식 (대형차량, 자전거 등)

4. **동역학 분류** (Dynamics-based)
   - 급가속/급감속 (Jerk)
   - 높은 측면 가속도

자세한 내용은 [시나리오 분류기 문서](./03_scenario_labeler.md)를 참조하세요.

### 라벨 카테고리

시스템은 다음과 같은 라벨 카테고리를 지원합니다:

- **속도 프로파일**: `low_magnitude_speed`, `medium_magnitude_speed`, `high_magnitude_speed`
- **정지 상황**: `stationary`, `stationary_in_traffic`
- **회전**: `starting_left_turn`, `starting_right_turn`, `starting_high_speed_turn`, `starting_low_speed_turn`
- **차선변경**: `changing_lane`, `changing_lane_to_left`, `changing_lane_to_right`
- **추종**: `following_lane_with_lead`, `following_lane_with_slow_lead`, `following_lane_without_lead`
- **근접**: `near_high_speed_vehicle`, `near_long_vehicle`, `near_multiple_vehicles`, `near_construction_zone_sign`, `near_trafficcone_on_driveable`, `near_barrier_on_driveable`, `behind_long_vehicle`, `behind_bike`, `near_multiple_pedestrians`
- **동역학**: `high_magnitude_jerk`, `high_lateral_acceleration`
- **맵 기반**: `on_stopline_traffic_light`, `on_stopline_stop_sign`, `on_stopline_crosswalk`, `on_intersection`, `on_traffic_light_intersection`, `on_all_way_stop_intersection`

### 맵 기반 분석

SQLite 기반 맵 데이터를 활용하여 정확한 위치 분석을 수행합니다:

- **차선 정보**: Lane, LaneConnector
- **도로 구조**: Roadblock, RoadblockConnector
- **교통 시설**: StopLine, Crosswalk
- **신호등**: RoadLight, PedestrianLight
- **공간 인덱싱**: STRtree를 활용한 효율적인 공간 쿼리

자세한 내용은 [맵 관리자 문서](./04_map_manager.md)를 참조하세요.

### 시각화

각 시나리오에 대해 PNG 이미지를 생성합니다:

- **Ego 중심 좌표계**: 60m x 60m 범위
- **차선 및 맵 요소**: 차선 폴리곤, 중심선, 횡단보도
- **차량 궤적**: 과거(실선), 미래(점선)
- **주변 객체**: 차량, 보행자, 자전거 등
- **라벨 정보**: 카테고리별 그룹화된 라벨 오버레이

자세한 내용은 [시각화 시스템 문서](./06_visualization.md)를 참조하세요.

## 시스템 아키텍처

시스템은 다음과 같은 주요 컴포넌트로 구성됩니다:

```mermaid
flowchart TD
    A[JSON Log File] --> B[JsonLogLoader]
    B -->|EgoState, TrackedObject 변환| C[MapManager]
    C -->|맵 데이터 로딩 및 쿼리| D[ScenarioWindow Extraction]
    D -->|101-epoch 윈도우 생성| E[ScenarioLabeler]
    E -->|4단계 분류 파이프라인| F[ScenarioExporter]
    E -->|4단계 분류 파이프라인| G[CustomScenario Visualizer]
    F -->|JSON 출력| H[(JSON 파일)]
    G -->|시각화| I[(PNG 이미지)]
```

자세한 내용은 [시스템 아키텍처 문서](./01_architecture.md)를 참조하세요.

## 데이터 구조

핵심 데이터 구조는 다음과 같습니다:

- **ScenarioWindow**: 101-epoch 시나리오 윈도우 (과거 40 + 현재 1 + 미래 60)
- **ScenarioLabel**: 라벨, 신뢰도, 카테고리 정보
- **LabeledScenario**: 라벨링 완료된 시나리오 (JSON 출력용)
- **LogEntry**: 단일 타임스탬프의 전체 상황 정보

자세한 내용은 [데이터 구조 문서](./02_data_structures.md)를 참조하세요.

## 처리 파이프라인

전체 처리 과정은 다음과 같습니다:

1. **로그 파일 로딩**: JSON 로그 파일을 읽고 파싱
2. **맵 초기화**: SQLite 맵 데이터베이스 로딩
3. **윈도우 추출**: 101-epoch 윈도우 생성 (과거 40 + 현재 1 + 미래 60)
4. **시나리오 분류**: 4단계 분류 파이프라인 실행
5. **JSON 출력**: 라벨링된 시나리오를 JSON 파일로 저장
6. **시각화**: 각 시나리오에 대한 PNG 이미지 생성
7. **통계 생성**: 라벨 분포 및 통계 정보 출력

자세한 내용은 [처리 파이프라인 문서](./08_pipeline.md)를 참조하세요.

## 설정

주요 설정은 `src/DefaultParams.py`에서 관리됩니다:

- **로그 파일 경로**: `LOG_FILE_NAME`
- **출력 디렉토리**: `OUTPUT_DIR_SCENARIO_FILE`, `OUTPUT_DIR_IMAGE_FILE`
- **처리 옵션**: `MAX_SCENARIOS`, `VISUALIZE`, `STEP_SIZE`
- **맵 설정**: `MAP_FILE_PATH`, `MAP_ORIGIN_X`, `MAP_ORIGIN_Y`

## 출력 결과

### JSON 출력

각 시나리오에 대해 다음 정보가 포함된 JSON 파일이 생성됩니다:

- 시나리오 ID 및 메타데이터
- Ego 차량 위치, 속도 정보
- 분류된 라벨 및 신뢰도
- 101-epoch 전체 관측 데이터
- 주변 차량/보행자 정보

자세한 내용은 [데이터 출력 문서](./07_export.md)를 참조하세요.

### 시각화 이미지

각 시나리오에 대해 PNG 이미지가 생성됩니다:

- Ego 중심 좌표계 (60m x 60m 범위)
- 차선 및 맵 요소 표시
- 차량 궤적 (과거: 실선, 미래: 점선)
- 분류된 라벨 오버레이

## 확장성

이 시스템은 다음과 같이 확장 가능합니다:

- 새로운 라벨 카테고리 추가
- 맞춤형 분류 규칙 구현
- 다양한 맵 형식 지원
- 실시간 분류 기능 추가

## 참고 자료

- [NuPlan 프레임워크](https://github.com/motional/nuplan-devkit)
- [Shapely](https://shapely.readthedocs.io/) - 공간 기하 연산
- [Matplotlib](https://matplotlib.org/) - 시각화

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

