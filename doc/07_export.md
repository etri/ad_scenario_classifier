# 데이터 출력 (ScenarioExporter)

ScenarioExporter는 라벨링된 시나리오를 JSON 형식으로 출력하는 컴포넌트입니다. 각 시나리오에 대한 완전한 정보를 포함한 JSON 파일을 생성합니다.

## 개요

ScenarioExporter는 다음과 같은 작업을 수행합니다:

- 라벨링된 시나리오를 JSON 형식으로 변환
- 101-epoch 전체 관측 데이터 포함
- 시나리오 메타데이터 생성
- 요약 파일 (scenarios_summary.json) 생성

## 클래스 구조

```python
class ScenarioExporter:
    @staticmethod
    def export_scenario(window: ScenarioWindow, output_path: str):
        # 단일 시나리오를 JSON 파일로 출력
    
    @staticmethod
    def export_batch(windows: List[ScenarioWindow], output_dir: str):
        # 여러 시나리오를 일괄 출력
```

## JSON 출력 구조

### 출력 데이터 흐름

```mermaid
flowchart TD
    A[ScenarioWindow] --> B[ScenarioExporter.export_scenario]
    B --> C[메타데이터 추출]
    B --> D[라벨 정보 추출]
    B --> E[통계 정보 계산]
    B --> F[Observation Data 생성]
    
    C --> C1[scenario_id]
    C --> C2[center_idx]
    C --> C3[center_timestamp]
    C --> C4[ego_position]
    C --> C5[ego_velocity]
    
    D --> D1[labels 리스트]
    D --> D2[label_details 리스트]
    
    E --> E1[num_agents]
    E --> E2[num_vehicles]
    E --> E3[num_pedestrians]
    E --> E4[confidence_mean]
    E --> E5[categories]
    
    F --> F1[ego_history<br/>40 프레임]
    F --> F2[ego_current<br/>1 프레임]
    F --> F3[ego_future<br/>60 프레임]
    F --> F4[agents_history]
    F --> F5[agents_current]
    F --> F6[agents_future]
    F --> F7[traffic_light_status]
    
    C1 --> G[LabeledScenario 객체]
    C2 --> G
    C3 --> G
    C4 --> G
    C5 --> G
    D1 --> G
    D2 --> G
    E1 --> G
    E2 --> G
    E3 --> G
    E4 --> G
    E5 --> G
    
    G --> H[JSON 딕셔너리 변환]
    F1 --> H
    F2 --> H
    F3 --> H
    F4 --> H
    F5 --> H
    F6 --> H
    F7 --> H
    
    H --> I[JSON 파일 저장]
    I --> J[scenario_XXXXXX.json]
```

### 배치 출력 프로세스

```mermaid
flowchart TD
    A[List[ScenarioWindow]] --> B[ScenarioExporter.export_batch]
    B --> C[출력 디렉토리 생성]
    C --> D[각 윈도우 반복]
    D --> E[export_scenario]
    E --> F[scenario_XXXXXX.json 생성]
    F --> G{모든 윈도우<br/>처리 완료?}
    G -->|아니오| D
    G -->|예| H[요약 파일 생성]
    H --> H1[scenarios_summary.json]
    H1 --> I[출력 완료]
```

## JSON 출력 형식

### 기본 구조

각 시나리오는 다음과 같은 JSON 구조로 출력됩니다:

```json
{
  "scenario_id": "scenario_000100",
  "center_idx": 100,
  "center_timestamp": 1234567890000000,
  "ego_position": {...},
  "ego_velocity": {...},
  "labels": [...],
  "label_details": [...],
  "num_agents": 5,
  "num_vehicles": 4,
  "num_pedestrians": 1,
  "confidence_mean": 0.895,
  "categories": ["speed_profile", "lane_change"],
  "observation_data": {...}
}
```

### 시나리오 메타데이터

#### scenario_id

시나리오 고유 식별자입니다.

```json
"scenario_id": "scenario_000100"
```

형식: `"scenario_{center_idx:06d}"`

#### center_idx

중심 프레임의 인덱스입니다.

```json
"center_idx": 100
```

#### center_timestamp

중심 프레임의 타임스탬프입니다 (마이크로초).

```json
"center_timestamp": 1234567890000000
```

### Ego 차량 정보

#### ego_position

Ego 차량의 현재 위치입니다.

```json
"ego_position": {
  "x": 230388.61912,
  "y": 424695.37128,
  "heading": 1.234
}
```

- **x**, **y**: UTM 좌표 (미터)
- **heading**: 헤딩 각도 (라디안)

#### ego_velocity

Ego 차량의 현재 속도입니다.

```json
"ego_velocity": {
  "vx": 5.5,
  "vy": 0.2,
  "magnitude": 5.504
}
```

- **vx**, **vy**: 속도 벡터 성분 (m/s)
- **magnitude**: 속도 크기 (m/s)

### 라벨 정보

#### labels

라벨명 리스트입니다.

```json
"labels": [
  "low_magnitude_speed",
  "changing_lane_to_left",
  "following_lane_with_lead"
]
```

#### label_details

각 라벨의 상세 정보입니다.

```json
"label_details": [
  {
    "label": "low_magnitude_speed",
    "confidence": 0.99,
    "category": "speed_profile"
  },
  {
    "label": "changing_lane_to_left",
    "confidence": 0.80,
    "category": "lane_change"
  },
  {
    "label": "following_lane_with_lead",
    "confidence": 0.85,
    "category": "following"
  }
]
```

각 라벨은 다음 정보를 포함합니다:
- **label**: 라벨명
- **confidence**: 신뢰도 (0.0~1.0)
- **category**: 카테고리

### 통계 정보

#### num_agents

현재 프레임의 총 에이전트 수입니다.

```json
"num_agents": 5
```

#### num_vehicles

현재 프레임의 차량 수입니다.

```json
"num_vehicles": 4
```

#### num_pedestrians

현재 프레임의 보행자 수입니다.

```json
"num_pedestrians": 1
```

#### confidence_mean

모든 라벨의 평균 신뢰도입니다.

```json
"confidence_mean": 0.895
```

#### categories

포함된 카테고리 리스트입니다 (중복 제거).

```json
"categories": [
  "speed_profile",
  "lane_change",
  "following"
]
```

## Observation Data

`observation_data`는 101-epoch 전체 관측 데이터를 포함합니다.

### 구조

```json
"observation_data": {
  "ego_history": [...],
  "ego_current": {...},
  "ego_future": [...],
  "agents_history": [...],
  "agents_current": [...],
  "agents_future": [...],
  "traffic_light_status": {...}
}
```

### Ego History

과거 40 프레임의 Ego 상태입니다.

```json
"ego_history": [
  {
    "timestamp": 1234567890000000,
    "position": {
      "x": 230388.61912,
      "y": 424695.37128,
      "heading": 1.234
    },
    "velocity": {
      "vx": 5.5,
      "vy": 0.2
    },
    "acceleration": {
      "ax": 0.1,
      "ay": 0.0
    }
  },
  ...
]
```

각 프레임은 다음 정보를 포함합니다:
- **timestamp**: 타임스탬프 (마이크로초)
- **position**: 위치 (x, y, heading)
- **velocity**: 속도 (vx, vy)
- **acceleration**: 가속도 (ax, ay)

### Ego Current

현재 프레임의 Ego 상태입니다.

```json
"ego_current": {
  "timestamp": 1234567890000000,
  "position": {
    "x": 230388.61912,
    "y": 424695.37128,
    "heading": 1.234
  },
  "velocity": {
    "vx": 5.5,
    "vy": 0.2
  },
  "acceleration": {
    "ax": 0.1,
    "ay": 0.0
  }
}
```

### Ego Future

미래 60 프레임의 Ego 상태입니다.

```json
"ego_future": [
  {
    "timestamp": 1234567890000000,
    "position": {...},
    "velocity": {...},
    "acceleration": {...}
  },
  ...
]
```

### Agents History

과거 각 프레임의 에이전트 리스트입니다.

```json
"agents_history": [
  [
    {
      "id": "agent_00000001",
      "type": "VEHICLE",
      "position": {
        "x": 230398.61912,
        "y": 424695.37128,
        "heading": 1.234
      },
      "velocity": {
        "vx": 6.0,
        "vy": 0.1
      },
      "box": {
        "length": 4.5,
        "width": 1.8,
        "height": 1.5
      }
    },
    ...
  ],
  ...
]
```

각 에이전트는 다음 정보를 포함합니다:
- **id**: 에이전트 고유 ID (track_token)
- **type**: 객체 타입 (VEHICLE, PEDESTRIAN, BICYCLE 등)
- **position**: 위치 (x, y, heading)
- **velocity**: 속도 (vx, vy) - Agent인 경우만
- **box**: 바운딩 박스 (length, width, height)

### Agents Current

현재 프레임의 에이전트 리스트입니다.

```json
"agents_current": [
  {
    "id": "agent_00000001",
    "type": "VEHICLE",
    "position": {...},
    "velocity": {...},
    "box": {...}
  },
  ...
]
```

### Agents Future

미래 각 프레임의 에이전트 리스트입니다.

```json
"agents_future": [
  [
    {
      "id": "agent_00000001",
      "type": "VEHICLE",
      "position": {...},
      "velocity": {...},
      "box": {...}
    },
    ...
  ],
  ...
]
```

### Traffic Light Status

신호등 상태 정보입니다.

```json
"traffic_light_status": {
  "status": "GO_STRAIGHT",
  "lane_connector_id": 12345,
  "timestamp": 1234567890000000
}
```

## 요약 파일

`scenarios_summary.json`은 모든 시나리오의 요약 정보를 포함합니다.

### 구조

```json
{
  "total_scenarios": 1000,
  "scenarios": [
    {
      "scenario_id": "scenario_000100",
      "center_idx": 100,
      "timestamp": 1234567890000000,
      "num_labels": 3,
      "labels": [
        "low_magnitude_speed",
        "changing_lane_to_left",
        "following_lane_with_lead"
      ]
    },
    ...
  ]
}
```

### 필드 설명

- **total_scenarios**: 총 시나리오 수
- **scenarios**: 시나리오 요약 리스트
  - **scenario_id**: 시나리오 ID
  - **center_idx**: 중심 프레임 인덱스
  - **timestamp**: 타임스탬프
  - **num_labels**: 라벨 수
  - **labels**: 라벨 리스트

## export_scenario 메서드

단일 시나리오를 JSON 파일로 출력합니다.

```python
@staticmethod
def export_scenario(window: ScenarioWindow, output_path: str):
    # Ego 상태 요약
    ego = window.ego_current
    speed = math.sqrt(
        ego.dynamic_car_state.rear_axle_velocity_2d.x**2 +
        ego.dynamic_car_state.rear_axle_velocity_2d.y**2
    )
    
    # 에이전트 통계
    agents = window.agents_current
    num_vehicles = sum(1 for a in agents 
                      if a.tracked_object_type == TrackedObjectType.VEHICLE)
    num_pedestrians = sum(1 for a in agents 
                         if a.tracked_object_type == TrackedObjectType.PEDESTRIAN)
    
    # LabeledScenario 생성
    labeled_scenario = LabeledScenario(
        scenario_id=f"scenario_{window.center_idx:06d}",
        center_idx=window.center_idx,
        center_timestamp=window.center_timestamp,
        ego_position={...},
        ego_velocity={...},
        labels=[label.label for label in window.labels],
        label_details=[...],
        num_agents=len(agents),
        num_vehicles=num_vehicles,
        num_pedestrians=num_pedestrians,
        confidence_mean=float(np.mean([label.confidence for label in window.labels])),
        categories=list(set(label.category for label in window.labels))
    )
    
    # observation_data 추가
    scenario_dict = asdict(labeled_scenario)
    scenario_dict['observation_data'] = {...}
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
```

## export_batch 메서드

여러 시나리오를 일괄 출력합니다.

```python
@staticmethod
def export_batch(windows: List[ScenarioWindow], output_dir: str):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 시나리오 출력
    for window in windows:
        filename = f"scenario_{window.center_idx:06d}.json"
        output_path = os.path.join(output_dir, filename)
        ScenarioExporter.export_scenario(window, output_path)
    
    # 요약 파일 생성
    summary = {
        'total_scenarios': len(windows),
        'scenarios': [
            {
                'scenario_id': f"scenario_{w.center_idx:06d}",
                'center_idx': w.center_idx,
                'timestamp': w.center_timestamp,
                'num_labels': len(w.labels),
                'labels': [label.label for label in w.labels]
            }
            for w in windows
        ]
    }
    
    summary_path = os.path.join(output_dir, 'scenarios_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
```

## 사용 예시

### 단일 시나리오 출력

```python
from ScenarioExporter import ScenarioExporter

# 시나리오 출력
output_path = "./labeled_scenarios/scenario_000100.json"
ScenarioExporter.export_scenario(window, output_path)
```

### 배치 출력

```python
# 여러 시나리오 일괄 출력
output_dir = "./labeled_scenarios"
ScenarioExporter.export_batch(windows, output_dir)
```

## 파일 명명 규칙

시나리오 파일은 다음과 같은 명명 규칙을 따릅니다:

```
scenario_{center_idx:06d}.json
```

예:
- `scenario_000100.json`
- `scenario_000200.json`
- `scenario_001000.json`

## JSON 인코딩

JSON 파일은 UTF-8 인코딩을 사용하며, 한글 등 비ASCII 문자도 올바르게 저장됩니다:

```python
json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
```

## 파일 크기

각 시나리오 JSON 파일의 크기는 다음과 같습니다:

- **메타데이터**: 약 1KB
- **Observation Data**: 약 50-200KB (에이전트 수에 따라 다름)
- **총 크기**: 약 50-200KB

101 프레임 × 평균 10개 에이전트 기준 약 100KB입니다.

## 성능 고려사항

### 출력 속도

- 단일 시나리오: 약 10-50ms
- 1000개 시나리오: 약 10-50초

### 메모리 사용

- 각 시나리오는 메모리에 유지
- 대용량 배치의 경우 메모리 부족 가능
- 스트리밍 방식으로 개선 가능

## 다음 단계

- [처리 파이프라인 문서](./08_pipeline.md) - 데이터 출력이 파이프라인에서 사용되는 방법
- [데이터 구조 문서](./02_data_structures.md) - LabeledScenario 구조 상세

