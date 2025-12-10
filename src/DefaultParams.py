"""
기본 파라미터 설정 모듈

시나리오 분류기 시스템의 실행 파라미터를 정의합니다.
데이터 파일 경로, 처리 옵션, 맵 설정 등을 포함합니다.

참조:
    - doc/01_architecture.md: 시스템 아키텍처 및 데이터 흐름
    - README.md: 프로젝트 개요 및 설정 방법
"""

# ============================================================================
# 데이터 파일 경로 설정
# ============================================================================

LOG_FILE_NAME = "dataset/observation_log_20250307-094118.json"
"""
입력 JSON 로그 파일 경로.

NuPlan 프레임워크 형식의 JSON 로그 파일을 지정합니다.
이 파일은 JsonLogLoader에 의해 로드되어 LogEntry 리스트로 변환됩니다.

형식:
    - 각 엔트리는 timestamp_us, ego_state, dynamic_agents를 포함
    - ego_state: x, y, heading, v_x, v_y, acc_x, acc_y 등
    - dynamic_agents: 주변 객체 리스트 (차량, 보행자 등)

참조:
    - JsonLogLoader: JSON 로그 로딩 및 파싱
    - doc/01_architecture.md: 데이터 로딩 레이어 설명
"""

OUTPUT_DIR_SCENARIO_FILE = "./labeled_scenarios"
"""
라벨링된 시나리오 JSON 출력 디렉토리.

ScenarioExporter가 생성한 JSON 파일들이 저장되는 디렉토리입니다.
각 시나리오는 개별 JSON 파일로 저장되며, scenarios_summary.json도 생성됩니다.

출력 파일 형식:
    - scenario_000100.json: 개별 시나리오 파일
    - scenarios_summary.json: 전체 시나리오 요약 파일

참조:
    - doc/02_data_structures.md: LabeledScenario 및 JSON 출력 형식
"""

OUTPUT_DIR_IMAGE_FILE = "./imgs"
"""
시각화 이미지 출력 디렉토리.

CustomScenarioVisualizer가 생성한 PNG 이미지들이 저장되는 디렉토리입니다.
각 시나리오는 ego 중심 좌표계로 시각화된 이미지로 저장됩니다.

출력 파일 형식:
    - scenario_000100.png: 시나리오 시각화 이미지

참조:
    - doc/06_visualization.md: 시각화 시스템 상세 설명
"""

# ============================================================================
# 처리 옵션 설정
# ============================================================================

MAX_SCENARIOS = -1
"""
처리할 최대 시나리오 수.

- -1: 전체 시나리오 처리 (모든 가능한 윈도우 생성)
- 양수: 해당 개수만 처리 (테스트나 샘플링용)

예:
    - MAX_SCENARIOS = 100: 처음 100개 시나리오만 처리
    - MAX_SCENARIOS = -1: 모든 시나리오 처리

참고:
    전체 로그가 수만 프레임인 경우, 모든 윈도우를 생성하면
    수천~수만 개의 시나리오가 생성될 수 있습니다.
"""

VISUALIZE = True
"""
시각화 이미지 생성 여부.

- True: 각 시나리오에 대해 PNG 이미지 생성
- False: JSON만 생성 (시각화 스킵, 처리 속도 향상)

시각화는 처리 시간을 크게 증가시킬 수 있으므로,
대량 처리 시 False로 설정하는 것을 권장합니다.

참조:
    - CustomScenarioVisualizer: 시각화 엔진
    - doc/06_visualization.md: 시각화 시스템 설명
"""

STEP_SIZE = 100
"""
시나리오 윈도우 생성 스텝 크기 (샘플링 간격).

윈도우 중심 인덱스 간의 간격을 지정합니다.
작은 값일수록 더 많은 시나리오가 생성되지만 처리 시간이 증가합니다.

예:
    - STEP_SIZE = 1: 모든 가능한 윈도우 생성 (가장 조밀한 샘플링)
    - STEP_SIZE = 100: 100 프레임마다 윈도우 생성 (약 5초 간격, 20Hz 기준)
    - STEP_SIZE = 200: 200 프레임마다 윈도우 생성 (약 10초 간격, 20Hz 기준)

윈도우 구조:
    - 각 윈도우는 101 프레임 (과거 40 + 현재 1 + 미래 60)
    - STEP_SIZE=1인 경우, 연속된 윈도우들이 1프레임씩 겹침
    - STEP_SIZE=100인 경우, 윈도우 간 100프레임 간격

참조:
    - doc/01_architecture.md: 윈도우 추출 레이어 및 샘플링 설명
"""

# ============================================================================
# 맵 설정
# ============================================================================

MAP_FILE_PATH = "src/map.sqlite"
"""
SQLite 맵 데이터베이스 파일 경로.

NuPlan 형식의 맵 데이터베이스 파일을 지정합니다.
MapManager가 이 파일을 로드하여 맵 쿼리 API를 제공합니다.

맵 데이터베이스 구조:
    - 차선 (Lane), 차선 연결부 (LaneConnector)
    - 도로 블록 (Roadblock), 정지선 (StopLine)
    - 횡단보도 (Crosswalk) 등

참조:
    - MapManager: 맵 데이터 관리자
    - doc/01_architecture.md: MapManager 역할 설명
"""

MAP_ORIGIN_X = 230388.61912
"""
맵 원점 X 좌표 (UTM 좌표계, 미터 단위).

JSON 로그의 로컬 좌표를 UTM 좌표계로 변환할 때 사용됩니다.
JsonLogLoader가 이 값을 사용하여 좌표계 변환을 수행합니다.

좌표계 변환:
    - 로컬 좌표 (JSON): (x, y) - 로그 파일 내 상대 좌표
    - UTM 좌표 (시스템): (x + MAP_ORIGIN_X, y + MAP_ORIGIN_Y)

참조:
    - JsonLogLoader.parse_ego_state(): 좌표계 변환 로직
    - doc/01_architecture.md: 좌표 변환 설명
"""

MAP_ORIGIN_Y = 424695.37128
"""
맵 원점 Y 좌표 (UTM 좌표계, 미터 단위).

JSON 로그의 로컬 좌표를 UTM 좌표계로 변환할 때 사용됩니다.
JsonLogLoader가 이 값을 사용하여 좌표계 변환을 수행합니다.

좌표계 변환:
    - 로컬 좌표 (JSON): (x, y) - 로그 파일 내 상대 좌표
    - UTM 좌표 (시스템): (x + MAP_ORIGIN_X, y + MAP_ORIGIN_Y)

참조:
    - JsonLogLoader.parse_ego_state(): 좌표계 변환 로직
    - doc/01_architecture.md: 좌표 변환 설명
"""
