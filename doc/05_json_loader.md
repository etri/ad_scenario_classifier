# JSON 로그 로더 (JsonLogLoader)

JsonLogLoader는 JSON 형식의 자율주행 로그 파일을 읽고 NuPlan 프레임워크의 데이터 구조로 변환하는 컴포넌트입니다.

## 개요

JsonLogLoader는 다음과 같은 작업을 수행합니다:

- JSON 로그 파일 로딩 및 파싱
- 좌표계 변환 (로컬 좌표 → 맵 원점 기준 UTM 좌표)
- EgoState 객체 생성
- TrackedObject 객체 생성 (차량, 보행자, 자전거 등)
- 신호등 상태 파싱

## 클래스 구조

```python
class JsonLogLoader:
    def __init__(self, log_file_path: str, mapOrigin):
        # 로그 파일 경로 및 맵 원점 설정
    
    def load(self) -> List[Dict[str, Any]]:
        # JSON 파일 로딩
    
    def parse_ego_state(self, ego_data: Dict, timestamp_us: int) -> EgoState:
        # Ego 상태 파싱
    
    def parse_traffic_light(self, ego_data: Dict, map_api) -> Tuple:
        # 신호등 상태 파싱
    
    def parse_dynamic_agent(self, agent_data: Dict, timestamp_us: int) -> TrackedObject:
        # 동적 에이전트 파싱
    
    def parse_log_entry(self, entry_data: Dict, map_api) -> LogEntry:
        # 로그 엔트리 파싱
    
    def get_parsed_entries(self, map_api) -> List[LogEntry]:
        # 모든 로그 엔트리 파싱
    
    def get_ego_states(self, parsed_entries) -> List[EgoState]:
        # Ego 상태 리스트 추출
    
    def get_dynamic_agents(self, parsed_entries) -> List[List[TrackedObject]]:
        # 동적 에이전트 리스트 추출
```

## JSON 로그 파일 형식

JSON 로그 파일은 다음과 같은 구조를 가집니다:

```json
[
  {
    "timestamp_us": 1234567890000000,
    "ego_state": {
      "x": 0.0,
      "y": 0.0,
      "heading": 90.0,
      "v_x": 5.5,
      "v_y": 0.2,
      "acc_x": 0.1,
      "acc_y": 0.0,
      "steer_angle": 0.0,
      "vehicle_traffic_light": 0
    },
    "dynamic_agents": [
      {
        "agent_id": 1,
        "category": 0,
        "x": 10.0,
        "y": 0.5,
        "heading": 90.0,
        "v_x": 6.0,
        "v_y": 0.1,
        "length": 4.5,
        "width": 1.8,
        "height": 1.5
      }
    ]
  }
]
```

### 필드 설명

#### 최상위 레벨

- **timestamp_us** (int): 타임스탬프 (마이크로초)
- **ego_state** (dict): Ego 차량 상태
- **dynamic_agents** (list): 주변 동적 객체 리스트

#### ego_state

- **x**, **y** (float): 로컬 좌표계 위치 (미터)
- **heading** (float): 헤딩 각도 (도, ENU 좌표계)
- **v_x**, **v_y** (float): 속도 벡터 (m/s)
- **acc_x**, **acc_y** (float): 가속도 벡터 (m/s²)
- **steer_angle** (float): 조향각 (라디안)
- **vehicle_traffic_light** (int): 신호등 상태 (0-6, 255=UNKNOWN)

#### dynamic_agents

- **agent_id** (int): 에이전트 고유 ID
- **category** (int): 객체 카테고리
  - 0: VEHICLE
  - 1: PEDESTRIAN
  - 2: BICYCLE
  - 3: TRAFFIC_CONE
  - 4: BARRIER
  - 5: CZONE_SIGN
  - 6: GENERIC_OBJECT
- **x**, **y** (float): 로컬 좌표계 위치 (미터)
- **heading** (float): 헤딩 각도 (도)
- **v_x**, **v_y** (float): 속도 벡터 (m/s, Agent인 경우)
- **length**, **width**, **height** (float): 바운딩 박스 크기 (미터)

## 좌표계 변환

### 로컬 좌표 → UTM 좌표

JSON 로그 파일의 좌표는 로컬 좌표계를 사용하며, 맵 원점을 기준으로 UTM 좌표로 변환됩니다:

```python
# 맵 원점 (DefaultParams.py에서 설정)
MAP_ORIGIN_X = 230388.61912
MAP_ORIGIN_Y = 424695.37128

# 변환
utm_x = local_x + MAP_ORIGIN_X
utm_y = local_y + MAP_ORIGIN_Y
```

### 헤딩 각도 변환

JSON 로그 파일의 헤딩은 ENU (East-North-Up) 좌표계를 사용하며, 유클리드 좌표계로 변환됩니다:

```python
# ENU → 유클리드 변환
# ENU: 0도 = 동쪽, 90도 = 북쪽
# 유클리드: 0도 = 동쪽, 반시계방향

heading_rad = np.deg2rad((90 - np.rad2deg(heading_deg)) % 360.0)
```

## EgoState 파싱

### parse_ego_state

JSON 데이터를 NuPlan의 EgoState 객체로 변환합니다:

```python
def parse_ego_state(self, ego_data: Dict[str, Any], timestamp_us: int) -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(
            ego_data['x'] + self.MAP_ORIGIN_X,    # UTM X 좌표
            ego_data['y'] + self.MAP_ORIGIN_Y,    # UTM Y 좌표
            np.deg2rad((90 - np.rad2deg(float(ego_data['heading']))) % 360.0)
        ),
        rear_axle_velocity_2d=StateVector2D(
            x=ego_data.get('v_x', 0.0),
            y=ego_data.get('v_y', 0.0)
        ),
        rear_axle_acceleration_2d=StateVector2D(
            x=ego_data.get('acc_x', 0.0),
            y=ego_data.get('acc_y', 0.0)
        ),
        tire_steering_angle=ego_data.get('steer_angle', 0.0),
        time_point=TimePoint(time_us=int(timestamp_us)),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True
    )
```

### 변환 과정

1. **위치 변환**: 로컬 좌표 + 맵 원점 → UTM 좌표
2. **헤딩 변환**: ENU 좌표계 → 유클리드 좌표계
3. **속도/가속도**: 그대로 사용 (벡터 성분)
4. **차량 파라미터**: Pacifica 기본값 사용

## TrackedObject 파싱

### parse_dynamic_agent

JSON 데이터를 NuPlan의 TrackedObject 객체로 변환합니다:

```python
def parse_dynamic_agent(
    self, 
    agent_data: Dict[str, Any], 
    timestamp_us: Optional[int] = None
) -> TrackedObject:
    # 포즈 생성
    pose = StateSE2(
        agent_data.get('x', 0.0) + self.MAP_ORIGIN_X,
        agent_data.get('y', 0.0) + self.MAP_ORIGIN_Y, 
        np.deg2rad(agent_data.get('heading', 0.0))
    )
    
    # 바운딩 박스 생성
    oriented_box = OrientedBox(
        pose, 
        width=agent_data.get('width', 0.0), 
        length=agent_data.get('length', 0.0), 
        height=agent_data.get('height', 0.0)
    )
    
    # 카테고리 매핑
    category = agent_data.get('category', 0)
    tracked_object_type = self.category_to_tracked_object_type.get(
        category, 
        TrackedObjectType.VEHICLE
    )
    
    # 메타데이터 생성
    agent_id = agent_data.get('agent_id', 0)
    track_token = f"agent_{agent_id:08d}"
    
    metadata = SceneObjectMetadata(
        token=f"token_{agent_id:08d}_{timestamp_us or 0}",
        track_token=track_token,
        track_id=get_unique_incremental_track_id(track_token),
        timestamp_us=timestamp_us or 0,
        category_name=tracked_object_type.name.lower(),
    )
    
    # Agent 또는 StaticObject 생성
    if tracked_object_type in AGENT_TYPES:
        return Agent(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            velocity=StateVector2D(
                agent_data.get('v_x', 0.0), 
                agent_data.get('v_y', 0.0)
            ),
            predictions=[],
            angular_velocity=0.0,
            metadata=metadata,
        )
    else:
        return StaticObject(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=metadata,
        )
```

### 카테고리 매핑

카테고리 번호를 TrackedObjectType으로 매핑합니다:

```python
category_to_tracked_object_type = {
    0: TrackedObjectType.VEHICLE,      # 차량
    1: TrackedObjectType.PEDESTRIAN,   # 보행자
    2: TrackedObjectType.BICYCLE,      # 자전거
    3: TrackedObjectType.TRAFFIC_CONE, # 콘
    4: TrackedObjectType.BARRIER,      # 장벽
    5: TrackedObjectType.CZONE_SIGN,   # 공사구역 표지판
    6: TrackedObjectType.GENERIC_OBJECT, # 기타
}
```

### Agent vs StaticObject

- **Agent**: 움직이는 객체 (VEHICLE, PEDESTRIAN, BICYCLE)
  - 속도 정보 포함
  - 예측 정보 포함 (현재는 빈 리스트)

- **StaticObject**: 정적 객체 (TRAFFIC_CONE, BARRIER, CZONE_SIGN)
  - 속도 정보 없음

## 신호등 파싱

### parse_traffic_light

Ego 위치 기반의 신호등 상태를 파싱합니다:

```python
def parse_traffic_light(self, ego_data: Dict[str, Any], map_api):
    x = ego_data.get('x', 0.0) + self.MAP_ORIGIN_X
    y = ego_data.get('y', 0.0) + self.MAP_ORIGIN_Y
    tl_status = ego_data.get('vehicle_traffic_light', TrafficLightStatusType.UNKNOWN)
    
    if tl_status == 255:  # 255 = UNKNOWN
        tl_status = TrafficLightStatusType.UNKNOWN
    
    # 가장 가까운 차선 ID 찾기
    ego_lane_id = int(map_api.get_nearest_lane((x, y))[0])
    
    return tl_status, ego_lane_id
```

### 신호등 상태 타입

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

## 로그 엔트리 파싱

### parse_log_entry

단일 로그 엔트리를 LogEntry 객체로 변환합니다:

```python
def parse_log_entry(self, entry_data: Dict[str, Any], map_api) -> LogEntry:
    timestamp_us = entry_data.get('timestamp_us', 0)
    
    # Ego 상태 파싱
    ego_state = self.parse_ego_state(
        entry_data.get('ego_state', {}), 
        timestamp_us
    )
    
    # 신호등 상태 파싱
    traffic_light_status, ego_lane_id = self.parse_traffic_light(
        entry_data.get('ego_state', {}), 
        map_api
    )
    
    # 동적 에이전트 파싱
    dynamic_agents = []
    for agent_data in entry_data.get('dynamic_agents', []):
        dynamic_agent = self.parse_dynamic_agent(agent_data, timestamp_us)
        dynamic_agents.append(dynamic_agent)
    
    return LogEntry(
        timestamp_us=timestamp_us,
        ego_state=ego_state,
        dynamic_agents=dynamic_agents,
        traffic_light_status=TrafficLightStatusData(
            traffic_light_status, 
            ego_lane_id, 
            timestamp_us
        )
    )
```

## 전체 파싱 프로세스

### 파싱 프로세스 흐름도

```mermaid
flowchart TD
    A[JSON 로그 파일] --> B[JsonLogLoader.load]
    B --> C[JSON 데이터 로딩]
    C --> D[get_parsed_entries]
    D --> E[각 엔트리 반복]
    E --> F[parse_log_entry]
    
    F --> G[parse_ego_state]
    F --> H[parse_dynamic_agent]
    F --> I[parse_traffic_light]
    
    G --> G1["로컬 좌표 -> UTM 좌표"]
    G1 --> G2["ENU 헤딩 -> 유클리드 헤딩"]
    G2 --> G3[EgoState 객체 생성]
    
    H --> H1["로컬 좌표 -> UTM 좌표"]
    H1 --> H2[카테고리 매핑]
    H2 --> H3{Agent 타입?}
    H3 -->|예| H4["Agent 객체 생성: 속도 포함"]
    H3 -->|아니오| H5[StaticObject 객체 생성]
    
    I --> I1["맵 쿼리: get_nearest_lane"]
    I1 --> I2[신호등 상태 파싱]
    I2 --> I3[TrafficLightStatusData 생성]
    
    G3 --> J[LogEntry 생성]
    H4 --> J
    H5 --> J
    I3 --> J
    
    J --> K["LogEntry 리스트"]
    K --> L["get_ego_states -> List[EgoState]"]
    K --> M["get_dynamic_agents -> List[List[TrackedObject]]"]
```

### get_parsed_entries

모든 로그 엔트리를 파싱합니다:

```python
def get_parsed_entries(self, map_api = None) -> List[LogEntry]:
    if self.data is None:
        self.load()  # JSON 파일 로딩
    
    parsed_entries = []
    for entry_data in (self.data or []):
        parsed_entry = self.parse_log_entry(entry_data, map_api)
        parsed_entries.append(parsed_entry)
    
    return parsed_entries
```

### 데이터 추출

파싱된 엔트리에서 특정 데이터를 추출합니다:

```python
# Ego 상태 리스트 추출
def get_ego_states(self, parsed_entries) -> List[EgoState]:
    return [entry.ego_state for entry in parsed_entries]

# 동적 에이전트 리스트 추출
def get_dynamic_agents(self, parsed_entries) -> List[List[TrackedObject]]:
    return [entry.dynamic_agents for entry in parsed_entries]
```

## 사용 예시

### 기본 사용

```python
from JsonLogLoader import JsonLogLoader
from MapManager import MapManager

# 맵 매니저 초기화 (신호등 파싱에 필요)
map_manager = MapManager("src/map.sqlite")
map_manager.initialize_all_layers()

# 로그 로더 초기화
map_origin = [230388.61912, 424695.37128]
log_loader = JsonLogLoader("dataset/log.json", map_origin)

# 로그 엔트리 파싱
parsed_entries = log_loader.get_parsed_entries(map_manager)

# 데이터 추출
ego_states = log_loader.get_ego_states(parsed_entries)
dynamic_agents = log_loader.get_dynamic_agents(parsed_entries)

print(f"총 {len(parsed_entries)}개의 로그 엔트리 파싱 완료")
```

### 개별 파싱

```python
# 단일 Ego 상태 파싱
ego_data = {
    "x": 0.0,
    "y": 0.0,
    "heading": 90.0,
    "v_x": 5.5,
    "v_y": 0.2
}
ego_state = log_loader.parse_ego_state(ego_data, 1234567890000000)

# 단일 에이전트 파싱
agent_data = {
    "agent_id": 1,
    "category": 0,
    "x": 10.0,
    "y": 0.5,
    "heading": 90.0,
    "v_x": 6.0,
    "v_y": 0.1,
    "length": 4.5,
    "width": 1.8,
    "height": 1.5
}
tracked_object = log_loader.parse_dynamic_agent(agent_data, 1234567890000000)
```

## 에러 처리

### 파일 없음

```python
try:
    log_loader.load()
except FileNotFoundError:
    print(f"로그 파일을 찾을 수 없습니다: {log_file_path}")
```

### JSON 파싱 에러

```python
try:
    log_loader.load()
except json.JSONDecodeError as e:
    print(f"JSON 파싱 에러: {e}")
```

### 데이터 검증

필수 필드가 없는 경우 기본값을 사용합니다:

- `v_x`, `v_y`: 0.0
- `acc_x`, `acc_y`: 0.0
- `steer_angle`: 0.0
- `heading`: 0.0
- `category`: 0 (VEHICLE)

## 성능 고려사항

### 메모리 사용

- 전체 JSON 파일을 메모리에 로딩
- 대용량 로그 파일의 경우 메모리 부족 가능
- 스트리밍 방식으로 개선 가능

### 파싱 속도

- 각 엔트리를 순차적으로 파싱
- 병렬 파싱으로 성능 향상 가능
- 맵 쿼리 (신호등 파싱)가 병목 가능

## 다음 단계

- [처리 파이프라인 문서](./08_pipeline.md) - 로그 로더가 파이프라인에서 사용되는 방법
- [데이터 구조 문서](./02_data_structures.md) - LogEntry 구조 상세

