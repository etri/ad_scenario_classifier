"""
데이터 타입 정의 모듈

이 모듈은 시나리오 분류기 시스템에서 사용되는 핵심 데이터 구조를 정의합니다.
주요 데이터 클래스:
- ScenarioLabel: 시나리오에 할당된 단일 라벨 정보
- ScenarioWindow: 101-epoch 시나리오 윈도우 (과거 40 + 현재 1 + 미래 60 프레임)
- LabeledScenario: 라벨링 완료된 시나리오 (JSON 출력용)

자세한 내용은 doc/02_data_structures.md를 참조하세요.
"""

# 데이터 클래스 및 타입 힌트
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# NuPlan 프레임워크에서 사용되는 차량 상태 및 객체 추적 관련
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject


@dataclass
class ScenarioLabel:
    """
    시나리오에 할당된 단일 라벨을 나타내는 데이터 클래스.
    
    ScenarioLabeler의 4단계 분류 파이프라인을 통해 생성되며, 각 라벨은
    시나리오의 특정 특성(속도, 행동, 상호작용, 동역학)을 나타냅니다.
    
    Attributes:
        label (str): 라벨의 고유 식별자
            예: "low_magnitude_speed", "changing_lane_to_left", "following_lane_with_lead"
            모든 가능한 라벨 목록은 doc/03_scenario_labeler.md 참조
            
        confidence (float): 라벨에 대한 신뢰도 점수
            범위: 0.0 ~ 1.0 (높을수록 더 확실한 라벨)
            라벨 타입에 따라 기본 신뢰도가 다름:
            - State-based: 0.95~0.99 (속도, 정지 등 명시적 상태)
            - Map-based: 0.90~0.95 (맵 기반 라벨)
            - Behavior-based: 0.80~0.85 (차선변경, 회전 등 궤적 기반)
            - Interaction-based: 0.80~0.90 (선행차 추종, 근접 등)
            - Dynamics-based: 0.95 (jerk, 측면 가속도 등)
            
        category (str): 라벨이 속한 카테고리
            가능한 값:
            - "speed_profile": 속도 프로파일 (저속/중속/고속)
            - "stationary": 정지 상황
            - "turning": 회전 (좌회전/우회전, 고속/저속)
            - "lane_change": 차선변경
            - "following": 선행차 추종
            - "proximity": 근접 객체
            - "dynamics": 동역학 특성
            - "other": 기타
            
        description (str): 라벨에 대한 상세 설명 (현재는 사용되지 않음, 향후 확장용)
    
    참조:
        - doc/02_data_structures.md: 데이터 구조 상세 설명
        - doc/03_scenario_labeler.md: 분류 알고리즘 및 라벨 목록
    """
    label: str          # 라벨명 (예: "low_magnitude_speed", "changing_lane_to_left")
    confidence: float   # 라벨에 대한 신뢰도 (0.0 ~ 1.0)
    category: str       # 라벨 카테고리 (예: "speed_profile", "lane_change")
    description: str = ""  # 라벨에 대한 상세 설명 (향후 확장용)


@dataclass
class ScenarioWindow:
    """
    101-epoch 시나리오 윈도우를 나타내는 핵심 데이터 구조.
    
    시나리오 분류는 시간적 맥락을 고려해야 하므로, 단일 프레임만으로는
    차선변경이나 회전 같은 행동을 정확히 판단하기 어렵습니다. 따라서
    과거와 미래 정보를 포함한 윈도우를 사용합니다.
    
    윈도우 구조 (20Hz 샘플링 기준):
    - 과거 (History): 40 프레임 (2초)
    - 현재 (Current): 1 프레임
    - 미래 (Future): 60 프레임 (3초)
    - 총 길이: 101 프레임 (5.05초)
    
    타임라인 예시:
        [과거 40] [현재 1] [미래 60]
        -40 ... -1  0   +1 ... +60
        i-40 ... i-1 i  i+1 ... i+60
    
    Attributes:
        center_idx (int): 중심 프레임의 인덱스
            로그 엔트리 리스트에서의 인덱스로, 윈도우의 기준점이 됩니다.
            윈도우 범위: [center_idx - 40, center_idx + 60]
            
        center_timestamp (int): 중심 프레임의 타임스탬프 (마이크로초)
            Unix epoch 기준 마이크로초 단위로, 시나리오의 시간적 위치를 식별합니다.
        
        ego_history (List[EgoState]): 과거 40 프레임의 ego 상태 리스트
            인덱스: [center_idx - 40, center_idx - 39, ..., center_idx - 1]
            시간: 약 2초 (20Hz 기준)
            용도: 궤적 분석, 행동 패턴 인식, 과거 행동 추적
            
        ego_current (EgoState): 현재 프레임의 ego 상태
            인덱스: center_idx
            용도: 현재 시점의 정확한 상태, 분류의 기준점
            
        ego_future (List[EgoState]): 미래 60 프레임의 ego 상태 리스트
            인덱스: [center_idx + 1, center_idx + 2, ..., center_idx + 60]
            시간: 약 3초 (20Hz 기준)
            용도: 미래 행동 예측, 의도 파악, 미래 궤적 분석
        
        agents_history (List[List[TrackedObject]]): 과거 각 프레임의 에이전트 리스트
            길이: 40 (과거 프레임 수)
            각 요소는 해당 프레임에서 관측된 모든 TrackedObject 리스트입니다.
            용도: 궤적 추적, 상호작용 분석, 과거 객체 위치 추적
            
        agents_current (List[TrackedObject]): 현재 프레임의 에이전트 리스트
            현재 시점에서 관측된 모든 주변 객체 (차량, 보행자, 자전거 등)
            용도: 현재 상황 분석, 근접 객체 판단
            
        agents_future (List[List[TrackedObject]]): 미래 각 프레임의 에이전트 리스트
            길이: 60 (미래 프레임 수)
            미래 예측 정보 (실제 관측 데이터, 예측이 아님)
            용도: 미래 상호작용 분석, 미래 궤적 추적
        
        traffic_light_status (Any): 신호등 상태 정보
            현재는 사용되지 않음 (향후 확장용)
            TrafficLightStatusData 타입을 사용할 수 있음
            
        map_context (Dict[str, Any]): 맵 컨텍스트 정보
            현재는 사용되지 않음 (향후 확장용)
            차선 정보, 교차로 정보 등을 저장할 수 있습니다.
            예: {"nearest_lane_id": 123, "intersection_id": 456}
        
        labels (List[ScenarioLabel]): 할당된 라벨 리스트
            초기에는 빈 리스트입니다.
            ScenarioLabeler.classify() 호출 후 채워집니다.
            하나의 시나리오는 여러 라벨을 가질 수 있습니다 (다중 라벨링).
    
    참조:
        - doc/01_architecture.md: 101-epoch 윈도우 개념 설명
        - doc/02_data_structures.md: 데이터 구조 상세 설명
    """
    center_idx: int         # 중심 프레임의 인덱스 (로그 엔트리 리스트 기준)
    center_timestamp: int   # 중심 프레임의 타임스탬프 (microsec)
    
    # Ego Vehicle 데이터
    ego_history: List[EgoState]    # 과거 40 프레임의 ego 상태 (약 2초, 20Hz 기준)
    ego_current: EgoState          # 현재 1 프레임의 ego 상태
    ego_future: List[EgoState]     # 미래 60 프레임의 ego 상태 (약 3초, 20Hz 기준)
    
    # 주변 에이전트(차량, 보행자 등) 데이터  
    agents_history: List[List[TrackedObject]]  # 과거 40 프레임의 에이전트 상태 리스트
    agents_current: List[TrackedObject]        # 현재 프레임의 에이전트 상태 리스트
    agents_future: List[List[TrackedObject]]   # 미래 60 프레임의 에이전트 상태 리스트
    
    # 신호등 상태 정보 (향후 확장용)
    traffic_light_status: Any = None
    
    # 맵 컨텍스트 정보 (차선, 교차로 등, 향후 확장용)
    map_context: Dict[str, Any] = field(default_factory=dict)
    
    # 할당된 라벨들 (ScenarioLabeler.classify() 호출 후 채워짐)
    labels: List[ScenarioLabel] = field(default_factory=list)


@dataclass  
class LabeledScenario:
    """
    라벨링이 완료된 시나리오를 JSON으로 출력하기 위한 데이터 구조.
    
    ScenarioWindow를 기반으로 생성되며, JSON 출력에 필요한 모든 정보를
    포함합니다. ScenarioExporter가 이 구조를 사용하여 JSON 파일을 생성합니다.
    
    Attributes:
        scenario_id (str): 시나리오 고유 식별자
            형식: "scenario_{center_idx:06d}"
            예: "scenario_000100" (center_idx=100인 경우)
        
        center_idx (int): 중심 프레임 인덱스
            원본 로그 엔트리 리스트에서의 인덱스
            
        center_timestamp (int): 중심 프레임 타임스탬프 (마이크로초)
            Unix epoch 기준 마이크로초 단위
        
        ego_position (Dict[str, float]): Ego 차량의 현재 위치
            {
                "x": float,      # UTM X 좌표 (미터)
                "y": float,      # UTM Y 좌표 (미터)
                "heading": float # 헤딩 각도 (라디안)
            }
        
        ego_velocity (Dict[str, float]): Ego 차량의 현재 속도
            {
                "vx": float,        # X 방향 속도 (m/s)
                "vy": float,        # Y 방향 속도 (m/s)
                "magnitude": float  # 속도 크기 (m/s)
            }
        
        labels (List[str]): 라벨명 리스트
            예: ["low_magnitude_speed", "changing_lane_to_left"]
            시나리오에 할당된 모든 라벨의 이름만 포함
        
        label_details (List[Dict[str, Any]]): 각 라벨의 상세 정보
            [
                {
                    "label": str,        # 라벨명
                    "confidence": float, # 신뢰도 (0.0~1.0)
                    "category": str      # 카테고리
                },
                ...
            ]
            labels와 동일한 순서로 정렬됨
        
        num_agents (int): 현재 프레임의 총 에이전트 수
            주변 객체의 총 개수 (차량, 보행자, 자전거 등 모두 포함)
        
        num_vehicles (int): 현재 프레임의 차량 수
            TrackedObjectType.VEHICLE 타입 객체의 개수
        
        num_pedestrians (int): 현재 프레임의 보행자 수
            TrackedObjectType.PEDESTRIAN 타입 객체의 개수
        
        confidence_mean (float): 모든 라벨의 평균 신뢰도
            모든 라벨의 confidence 값의 평균
            시나리오 전체의 신뢰도를 나타냄
        
        categories (List[str]): 포함된 카테고리 리스트 (중복 제거)
            예: ["speed_profile", "lane_change"]
            시나리오에 할당된 라벨들의 고유 카테고리 목록
    
    참조:
        - doc/02_data_structures.md: 데이터 구조 및 JSON 출력 형식 상세
        - doc/07_export.md: 데이터 출력 문서 (존재하는 경우)
    """
    scenario_id: str        # 시나리오 고유 ID (형식: "scenario_{center_idx:06d}")
    center_idx: int         # 중심 프레임 인덱스
    center_timestamp: int   # 중심 프레임 타임스탬프 (microsec)
    
    # 현재 상태 요약 정보
    ego_position: Dict[str, float]  # ego 위치 (x, y, heading) - UTM 좌표계
    ego_velocity: Dict[str, float]  # ego 속도 (vx, vy, magnitude) - m/s
    
    # 라벨 정보
    labels: List[str]                    # 라벨명 리스트
    label_details: List[Dict[str, Any]]  # 라벨 상세 정보 (신뢰도, 카테고리 포함)
    
    # 통계 정보
    num_agents: int       # 총 에이전트 수 (현재 프레임 기준)
    num_vehicles: int     # 차량 수 (현재 프레임 기준)
    num_pedestrians: int  # 보행자 수 (현재 프레임 기준)
    
    # 메타데이터
    confidence_mean: float  # 평균 신뢰도 (모든 라벨의 confidence 평균)
    categories: List[str]   # 카테고리 리스트 (중복 제거된 고유 카테고리)
