# 데이터 구조

이 문서는 시나리오 분류기 시스템에서 사용되는 핵심 데이터 구조에 대한 상세 설명을 제공합니다.

## 개요

시스템은 다음과 같은 주요 데이터 구조를 사용합니다:

1. **ScenarioLabel**: 단일 라벨 정보
2. **ScenarioWindow**: 101-epoch 시나리오 윈도우
3. **LabeledScenario**: 라벨링 완료된 시나리오 (JSON 출력용)
4. **LogEntry**: 단일 타임스탬프의 전체 상황 정보
5. **TrafficLightStatusData**: 신호등 상태 정보

## ScenarioLabel

시나리오에 할당된 단일 라벨을 나타내는 데이터 클래스입니다.

### 데이터 구조

```mermaid
classDiagram
    class ScenarioLabel {
        +str label
        +float confidence
        +str category
        +str description
    }
    note for ScenarioLabel "label: low_magnitude_speed\nconfidence: 0.0~1.0\ncategory: speed_profile"
```

### 정의

```python
@dataclass
class ScenarioLabel:
    label: str          # 라벨명 (예: "changing_lane", "stationary")
    confidence: float   # 라벨에 대한 신뢰도 (0.0 ~ 1.0)
    category: str       # 라벨 카테고리 (예: "lane_change", "stationary")
    description: str = ""  # 라벨에 대한 상세 설명 (선택적)
```

### 필드 설명

- **label** (str): 라벨의 고유 식별자
  - 예: `"low_magnitude_speed"`, `"changing_lane_to_left"`, `"following_lane_with_lead"`
  - 모든 가능한 라벨 목록은 [시나리오 분류기 문서](./03_scenario_labeler.md) 참조

- **confidence** (float): 라벨의 신뢰도 점수
  - 범위: 0.0 ~ 1.0
  - 높은 값일수록 더 확실한 라벨
  - 라벨 타입에 따라 기본 신뢰도가 다름:
    - State-based: 0.95~0.99
    - Map-based: 0.90~0.95
    - Behavior-based: 0.80~0.85
    - Interaction-based: 0.80~0.90
    - Dynamics-based: 0.95

- **category** (str): 라벨이 속한 카테고리
  - 가능한 값: `"speed_profile"`, `"stationary"`, `"turning"`, `"lane_change"`, `"following"`, `"proximity"`, `"dynamics"`, `"other"`

- **description** (str): 라벨에 대한 설명 (현재는 사용되지 않음)

### 사용 예시

```python
label = ScenarioLabel(
    label="changing_lane_to_left",
    confidence=0.80,
    category="lane_change"
)
```

## ScenarioWindow

101-epoch 시나리오 윈도우를 나타내는 핵심 데이터 구조입니다. 과거, 현재, 미래의 ego 상태와 주변 객체 정보를 포함합니다.

### 정의

```python
@dataclass
class ScenarioWindow:
    center_idx: int         # 중심 프레임의 인덱스
    center_timestamp: int   # 중심 프레임의 타임스탬프 (microsec)
    
    # Ego Vehicle 데이터
    ego_history: List[EgoState]    # 과거 ego 상태 리스트 (40 프레임)
    ego_current: EgoState          # 현재 ego 상태 (1 프레임)
    ego_future: List[EgoState]     # 미래 ego 상태 리스트 (60 프레임)
    
    # 주변 에이전트(차량, 보행자 등) 데이터  
    agents_history: List[List[TrackedObject]]  # 과거 에이전트 상태 리스트
    agents_current: List[TrackedObject]        # 현재 에이전트 상태 리스트
    agents_future: List[List[TrackedObject]]   # 미래 에이전트 상태 리스트
    
    # 신호등 상태 정보
    traffic_light_status: Any = None
    
    # 맵 컨텍스트 정보 (차선, 교차로 등)
    map_context: Dict[str, Any] = field(default_factory=dict)
    
    # 할당된 라벨들
    labels: List[ScenarioLabel] = field(default_factory=list)
```

### 필드 설명

#### 시간 정보

- **center_idx** (int): 중심 프레임의 인덱스
  - 로그 엔트리 리스트에서의 인덱스
  - 윈도우의 기준점

- **center_timestamp** (int): 중심 프레임의 타임스탬프
  - 마이크로초 단위
  - 시나리오의 시간적 위치 식별

#### Ego Vehicle 데이터

- **ego_history** (List[EgoState]): 과거 40 프레임의 ego 상태
  - 인덱스: `[center_idx - 40, center_idx - 39, ..., center_idx - 1]`
  - 시간: 약 2초 (20Hz 기준)
  - 궤적 분석, 행동 패턴 인식에 사용

- **ego_current** (EgoState): 현재 프레임의 ego 상태
  - 인덱스: `center_idx`
  - 현재 시점의 정확한 상태

- **ego_future** (List[EgoState]): 미래 60 프레임의 ego 상태
  - 인덱스: `[center_idx + 1, center_idx + 2, ..., center_idx + 60]`
  - 시간: 약 3초 (20Hz 기준)
  - 미래 행동 예측, 의도 파악에 사용

#### 주변 에이전트 데이터

- **agents_history** (List[List[TrackedObject]]): 과거 각 프레임의 에이전트 리스트
  - 길이: 40 (과거 프레임 수)
  - 각 요소는 해당 프레임에서 관측된 모든 TrackedObject 리스트
  - 궤적 추적, 상호작용 분석에 사용

- **agents_current** (List[TrackedObject]): 현재 프레임의 에이전트 리스트
  - 현재 시점에서 관측된 모든 주변 객체

- **agents_future** (List[List[TrackedObject]]): 미래 각 프레임의 에이전트 리스트
  - 길이: 60 (미래 프레임 수)
  - 미래 예측 정보 (실제 관측 데이터)

#### 기타 정보

- **traffic_light_status**: 신호등 상태 정보
  - 현재는 사용되지 않음 (향후 확장용)

- **map_context** (Dict[str, Any]): 맵 컨텍스트 정보
  - 현재는 사용되지 않음 (향후 확장용)
  - 차선 정보, 교차로 정보 등을 저장할 수 있음

- **labels** (List[ScenarioLabel]): 할당된 라벨 리스트
  - 초기에는 빈 리스트
  - `ScenarioLabeler.classify()` 호출 후 채워짐

### EgoState 구조

EgoState는 NuPlan 프레임워크의 데이터 구조로, 다음 정보를 포함합니다:

- **rear_axle**: 후축 위치 및 헤딩
  - `x`, `y`: 위치 (UTM 좌표계, 미터)
  - `heading`: 헤딩 각도 (라디안)
  - `array`: numpy 배열 `[x, y, heading]`

- **dynamic_car_state**: 동적 상태
  - `rear_axle_velocity_2d`: 속도 벡터 (m/s)
    - `x`, `y`: 속도 성분
  - `rear_axle_acceleration_2d`: 가속도 벡터 (m/s²)
    - `x`, `y`: 가속도 성분

- **car_footprint**: 차량 외곽선
  - `geometry`: Shapely Polygon 객체

- **timestamp_us**: 타임스탬프 (마이크로초)

### TrackedObject 구조

TrackedObject는 NuPlan 프레임워크의 데이터 구조로, 주변 객체를 나타냅니다:

- **tracked_object_type**: 객체 타입
  - `TrackedObjectType.VEHICLE`: 차량
  - `TrackedObjectType.PEDESTRIAN`: 보행자
  - `TrackedObjectType.BICYCLE`: 자전거
  - `TrackedObjectType.TRAFFIC_CONE`: 콘
  - `TrackedObjectType.BARRIER`: 장벽
  - `TrackedObjectType.CZONE_SIGN`: 공사구역 표지판

- **center**: 중심 위치 및 헤딩
  - `x`, `y`: 위치 (UTM 좌표계, 미터)
  - `heading`: 헤딩 각도 (라디안)

- **box**: 바운딩 박스
  - `length`, `width`, `height`: 크기 (미터)
  - `geometry`: Shapely Polygon 객체

- **velocity**: 속도 벡터 (Agent인 경우)
  - `x`, `y`: 속도 성분 (m/s)

- **track_token**: 객체 추적 토큰 (고유 ID)

### 사용 예시

```python
window = ScenarioWindow(
    center_idx=100,
    center_timestamp=1234567890000000,
    ego_history=[...],  # 40개의 EgoState
    ego_current=ego_state_100,
    ego_future=[...],   # 60개의 EgoState
    agents_history=[...],  # 40개의 TrackedObject 리스트
    agents_current=[...],  # 현재 프레임의 TrackedObject 리스트
    agents_future=[...],   # 60개의 TrackedObject 리스트
    labels=[]  # 초기에는 빈 리스트
)

# 분류 후
labeler.classify(window)
# window.labels가 채워짐
```

## LabeledScenario

라벨링이 완료된 시나리오를 JSON으로 출력하기 위한 데이터 구조입니다.

### 데이터 구조

```mermaid
classDiagram
    class LabeledScenario {
        +str scenario_id
        +int center_idx
        +int center_timestamp
        +Dict ego_position
        +Dict ego_velocity
        +List~str~ labels
        +List~Dict~ label_details
        +int num_agents
        +int num_vehicles
        +int num_pedestrians
        +float confidence_mean
        +List~str~ categories
        +Dict observation_data
    }
    
    class ScenarioWindow {
        +int center_idx
        +List~ScenarioLabel~ labels
    }
    
    ScenarioWindow --> LabeledScenario : export_scenario
```

### 정의

```python
@dataclass  
class LabeledScenario:
    scenario_id: str        # 시나리오 고유 ID
    center_idx: int         # 중심 프레임 인덱스
    center_timestamp: int   # 중심 프레임 타임스탬프
    
    # 현재 상태 요약 정보
    ego_position: Dict[str, float]  # ego 위치 (x, y, heading)
    ego_velocity: Dict[str, float]  # ego 속도 (vx, vy, magnitude)
    
    # 라벨 정보
    labels: List[str]                    # 라벨명 리스트
    label_details: List[Dict[str, Any]]  # 라벨 상세 정보 (신뢰도, 카테고리 포함)
    
    # 통계 정보
    num_agents: int       # 총 에이전트 수
    num_vehicles: int     # 차량 수
    num_pedestrians: int  # 보행자 수
    
    # 메타데이터
    confidence_mean: float  # 평균 신뢰도
    categories: List[str]   # 카테고리 리스트
```

### 필드 설명

- **scenario_id** (str): 시나리오 고유 식별자
  - 형식: `"scenario_{center_idx:06d}"`
  - 예: `"scenario_000100"`

- **ego_position** (Dict): Ego 차량의 현재 위치
  ```python
  {
      "x": 230388.61912,      # UTM X 좌표 (미터)
      "y": 424695.37128,      # UTM Y 좌표 (미터)
      "heading": 1.234        # 헤딩 각도 (라디안)
  }
  ```

- **ego_velocity** (Dict): Ego 차량의 현재 속도
  ```python
  {
      "vx": 5.5,              # X 방향 속도 (m/s)
      "vy": 0.2,              # Y 방향 속도 (m/s)
      "magnitude": 5.504      # 속도 크기 (m/s)
  }
  ```

- **labels** (List[str]): 라벨명 리스트
  - 예: `["low_magnitude_speed", "changing_lane_to_left"]`

- **label_details** (List[Dict]): 각 라벨의 상세 정보
  ```python
  [
      {
          "label": "low_magnitude_speed",
          "confidence": 0.99,
          "category": "speed_profile"
      },
      ...
  ]
  ```

- **num_agents** (int): 현재 프레임의 총 에이전트 수
- **num_vehicles** (int): 현재 프레임의 차량 수
- **num_pedestrians** (int): 현재 프레임의 보행자 수

- **confidence_mean** (float): 모든 라벨의 평균 신뢰도
- **categories** (List[str]): 포함된 카테고리 리스트 (중복 제거)

## LogEntry

단일 타임스탬프에서의 전체 상황 정보를 나타내는 데이터 구조입니다.

### 정의

```python
@dataclass
class LogEntry:
    timestamp_us: int                            # 타임스탬프 (microsec)
    ego_state: EgoState                          # ego 상태 정보
    dynamic_agents: List[TrackedObject]          # 주변 동적 객체들
    traffic_light_status: TrafficLightStatusData # 신호등 상태 정보
```

### 필드 설명

- **timestamp_us** (int): 타임스탬프 (마이크로초)
  - Unix epoch 기준 마이크로초

- **ego_state** (EgoState): Ego 차량의 상태
  - 위치, 속도, 가속도, 헤딩 등

- **dynamic_agents** (List[TrackedObject]): 주변 동적 객체 리스트
  - 차량, 보행자, 자전거 등
  - 정적 객체(콘, 장벽 등)도 포함될 수 있음

- **traffic_light_status** (TrafficLightStatusData): 신호등 상태 정보
  - 자세한 내용은 아래 참조

## TrafficLightStatusData

신호등 상태 정보를 나타내는 데이터 구조입니다.

### 정의

```python
@dataclass
class TrafficLightStatusData:
    status: TrafficLightStatusType  # 신호등 상태
    lane_connector_id: int          # 신호등이 속한 차선 연결부 ID
    timestamp: int                  # 타임스탬프 (마이크로초)
```

### TrafficLightStatusType

신호등 상태를 나타내는 열거형:

```python
class TrafficLightStatusType(IntEnum):
    UNKNOWN = 6                     # 알 수 없음
    GO_STRAIGHT = 0                 # 직진 가능
    GO_STRAIGHT_AND_TURNLEFT = 1    # 직진 및 좌회전 가능
    STOP = 2                        # 정지
    STOP_AND_TURNLEFT = 3           # 정지 및 좌회전 가능
    STOP_AND_WARNING = 4            # 정지 및 경고
    WARNING = 5                     # 경고
```

### 필드 설명

- **status** (TrafficLightStatusType): 신호등 상태
- **lane_connector_id** (int): 신호등이 속한 차선 연결부 ID
  - 맵 데이터베이스의 차선 ID
- **timestamp** (int): 타임스탬프 (마이크로초)

## 데이터 변환 흐름

### 1. JSON → LogEntry

```mermaid
flowchart TD
    A[JSON 로그 파일] --> B[JsonLogLoader.parse_log_entry]
    B --> C["parse_ego_state -> EgoState"]
    B --> D["parse_dynamic_agent -> TrackedObject"]
    B --> E["parse_traffic_light -> TrafficLightStatusData"]
    C --> F[LogEntry]
    D --> F
    E --> F
```

### 2. LogEntry 리스트 → ScenarioWindow

```mermaid
flowchart TD
    A["LogEntry 리스트"] --> B[윈도우 추출]
    B --> C["ego_history: List[EgoState]"]
    B --> D["ego_current: EgoState"]
    B --> E["ego_future: List[EgoState]"]
    B --> F["agents_history: List[List[TrackedObject]]"]
    B --> G["agents_current: List[TrackedObject]"]
    B --> H["agents_future: List[List[TrackedObject]]"]
    C --> I[ScenarioWindow]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
```

### 3. ScenarioWindow → LabeledScenario

```mermaid
flowchart TD
    A[ScenarioWindow] --> B[ScenarioExporter.export_scenario]
    B --> C[메타데이터 추출]
    B --> D[라벨 정보 추출]
    B --> E[통계 정보 계산]
    B --> F["observation_data 생성: 101-epoch 전체 데이터"]
    C --> G["LabeledScenario / JSON 출력용"]
    D --> G
    E --> G
    F --> G
```

## JSON 출력 형식

LabeledScenario는 다음과 같은 JSON 구조로 출력됩니다:

```json
{
  "scenario_id": "scenario_000100",
  "center_idx": 100,
  "center_timestamp": 1234567890000000,
  "ego_position": {
    "x": 230388.61912,
    "y": 424695.37128,
    "heading": 1.234
  },
  "ego_velocity": {
    "vx": 5.5,
    "vy": 0.2,
    "magnitude": 5.504
  },
  "labels": ["low_magnitude_speed", "changing_lane_to_left"],
  "label_details": [
    {
      "label": "low_magnitude_speed",
      "confidence": 0.99,
      "category": "speed_profile"
    }
  ],
  "num_agents": 5,
  "num_vehicles": 4,
  "num_pedestrians": 1,
  "confidence_mean": 0.895,
  "categories": ["speed_profile", "lane_change"],
  "observation_data": {
    "ego_history": [...],
    "ego_current": {...},
    "ego_future": [...],
    "agents_history": [...],
    "agents_current": [...],
    "agents_future": [...],
    "traffic_light_status": {...}
  }
}
```

자세한 JSON 출력 형식은 [데이터 출력 문서](./07_export.md)를 참조하세요.

## 다음 단계

- [시나리오 분류기 문서](./03_scenario_labeler.md) - 분류 알고리즘에서 데이터 구조 사용
- [데이터 출력 문서](./07_export.md) - JSON 출력 형식 상세

