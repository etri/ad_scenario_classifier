# 시나리오 분류기 (ScenarioLabeler)

시나리오 분류기는 Rule-based 4단계 파이프라인을 통해 자율주행 데이터에서 다양한 주행 시나리오를 자동으로 라벨링하는 핵심 컴포넌트입니다.

## 개요

`ScenarioLabeler` 클래스는 101-epoch 시나리오 윈도우를 입력받아 다음과 같은 4단계 분류 파이프라인을 수행합니다:

1. **State-based Classification**: 명시적 상태 분류 (속도, 위치, 근접)
2. **Behavior-based Classification**: 궤적 기반 행동 분류
3. **Interaction-based Classification**: 상호작용 기반 분류
4. **Dynamics-based Classification**: 동역학 분류

각 단계는 독립적으로 실행되며, 최종적으로 모든 라벨이 통합되어 시나리오에 할당됩니다.

## 클래스 구조

```python
class ScenarioLabeler:
    def __init__(self, map_manager=None, hz: float = 20.0):
        # 초기화: 맵 매니저, 샘플링 주파수 설정
    
    def classify(self, window: ScenarioWindow) -> ScenarioWindow:
        # 4단계 분류 파이프라인 실행
    
    # 4단계 분류 메서드
    def _classify_explicit_states(self, window) -> Set[str]
    def _classify_behaviors(self, window) -> Set[str]
    def _classify_interactions(self, window) -> Set[str]
    def _classify_dynamics(self, window) -> Set[str]
    
    # 헬퍼 메서드들
    def _get_speed(self, ego: EgoState) -> float
    def _detect_turning(self, ...) -> Set[str]
    def _detect_lane_change(self, ...) -> Set[str]
    # ... 기타 헬퍼 메서드들
```

## 1단계: State-based Classification

명시적 상태 기반 분류는 현재 시점의 속도, 위치, 근접 객체 정보를 기반으로 라벨을 할당합니다.

### 속도 프로파일 분류

속도 크기에 따라 세 가지 라벨을 할당합니다:

```python
speed = self._get_speed(ego)  # 속도 크기 계산 (m/s)

if speed < SPEED_LOW_THRESHOLD:        # < 2.78 m/s (10 km/h)
    labels.add("low_magnitude_speed")
elif speed < SPEED_MEDIUM_THRESHOLD:   # < 11.11 m/s (40 km/h)
    labels.add("medium_magnitude_speed")
else:                                  # >= 11.11 m/s
    labels.add("high_magnitude_speed")
```

**임계값**:
- `SPEED_LOW_THRESHOLD = 2.78 m/s` (10 km/h)
- `SPEED_MEDIUM_THRESHOLD = 11.11 m/s` (40 km/h)

**신뢰도**: 0.99 (매우 높음)

### 정지 상태 감지

일정 시간 동안 속도가 임계값 이하인 경우 정지 상태로 판단합니다:

```python
if self._is_stationary(window.ego_history[-10:] + [ego]):
    labels.add("stationary")
    
    # 주변 차량이 많으면 교통체증 정지
    nearby_vehicles = self._count_nearby_vehicles(ego, agents, radius=30.0)
    if nearby_vehicles > 5:
        labels.add("stationary_in_traffic")
```

**임계값**:
- `STATIONARY_SPEED_THRESHOLD = 0.1 m/s`
- `STATIONARY_DURATION = 0.5 seconds` (10 프레임, 20Hz 기준)
- 교통체증 판단: 반경 30m 내 차량 5대 이상

**신뢰도**: 
- `stationary`: 0.98
- `stationary_in_traffic`: 0.95

### 근접 객체 분류

주변 객체와의 거리를 기반으로 근접 라벨을 할당합니다:

```python
def _classify_proximity(self, ego: EgoState, agents: List[TrackedObject]) -> Set[str]:
    labels = set()
    ego_pos = np.array([ego.rear_axle.x, ego.rear_axle.y])
    
    for agent in agents:
        agent_pos = np.array([agent.center.x, agent.center.y])
        distance = np.linalg.norm(ego_pos - agent_pos)
        
        if agent.tracked_object_type == TrackedObjectType.VEHICLE:
            speed = self._get_agent_speed(agent)
            # 고속 차량 근접
            if speed > HIGH_SPEED_VEHICLE_THRESHOLD and distance < 50.0:
                labels.add("near_high_speed_vehicle")
            # 대형 차량 근접
            if self._is_long_vehicle(agent) and distance < 30.0:
                labels.add("near_long_vehicle")
        
        elif agent.tracked_object_type == TrackedObjectType.CZONE_SIGN:
            if distance < CONSTRUCTION_ZONE_DISTANCE:
                labels.add("near_construction_zone_sign")
        
        elif agent.tracked_object_type == TrackedObjectType.TRAFFIC_CONE:
            if distance < TRAFFIC_CONE_DISTANCE:
                labels.add("near_trafficcone_on_driveable")
        
        elif agent.tracked_object_type == TrackedObjectType.BARRIER:
            if distance < TRAFFIC_CONE_DISTANCE:
                labels.add("near_barrier_on_driveable")
    
    return labels
```

**임계값**:
- `HIGH_SPEED_VEHICLE_THRESHOLD = 20.0 m/s` (72 km/h)
- `CONSTRUCTION_ZONE_DISTANCE = 20.0 m`
- `TRAFFIC_CONE_DISTANCE = 10.0 m`
- `LONG_VEHICLE_LENGTH = 8.0 m` (대형 차량 판단 기준)

**신뢰도**:
- `near_high_speed_vehicle`: 0.85
- `near_long_vehicle`: 0.90
- `near_construction_zone_sign`: 0.90
- `near_trafficcone_on_driveable`: 0.85
- `near_barrier_on_driveable`: 0.85

## 2단계: Behavior-based Classification

궤적 기반 행동 분류는 과거와 미래의 궤적을 분석하여 차선변경, 회전 등의 행동을 감지합니다.

### 회전 감지

헤딩 각도 변화를 기반으로 회전을 감지합니다:

```python
def _detect_turning(self, ego_history, ego_current, ego_future) -> Set[str]:
    labels = set()
    
    if len(ego_history) < 5 or len(ego_future) < 5:
        return labels
    
    # 과거와 미래의 헤딩 각도 비교
    past_heading = ego_history[0].rear_axle.heading
    future_heading = ego_future[-1].rear_axle.heading
    
    heading_change_rad = self._normalize_angle(future_heading - past_heading)
    heading_change_deg = math.degrees(heading_change_rad)
    
    if abs(heading_change_deg) > TURN_HEADING_THRESHOLD:
        # 좌회전 또는 우회전
        if heading_change_deg > 0:
            labels.add("starting_left_turn")
        else:
            labels.add("starting_right_turn")
        
        # 고속/저속 회전 구분
        speed = self._get_speed(ego_current)
        if speed > HIGH_SPEED_TURN_THRESHOLD:
            labels.add("starting_high_speed_turn")
        else:
            labels.add("starting_low_speed_turn")
    
    return labels
```

**임계값**:
- `TURN_HEADING_THRESHOLD = 15.0 degrees`
- `HIGH_SPEED_TURN_THRESHOLD = 8.0 m/s` (28.8 km/h)

**신뢰도**: 0.80~0.85

### 차선변경 감지

횡방향 변위를 기반으로 차선변경을 감지합니다:

```python
def _detect_lane_change(self, ego_history, ego_current, ego_future) -> Set[str]:
    labels = set()
    
    if len(ego_history) < 10 or len(ego_future) < 10:
        return labels
    
    # 과거와 미래의 위치 비교
    past_positions = np.array([[s.rear_axle.x, s.rear_axle.y] 
                               for s in ego_history[-10:]])
    future_positions = np.array([[s.rear_axle.x, s.rear_axle.y] 
                                 for s in ego_future[:10]])
    current_pos = np.array([ego_current.rear_axle.x, ego_current.rear_axle.y])
    
    # 미래 방향 벡터
    future_vector = future_positions[-1] - current_pos
    
    # 과거 헤딩에 수직인 방향 벡터
    past_heading = ego_history[-10].rear_axle.heading
    perpendicular = np.array([-math.sin(past_heading), math.cos(past_heading)])
    
    # 횡방향 변위 계산
    lateral_displacement = np.dot(future_vector, perpendicular)
    
    if abs(lateral_displacement) > 1.5:  # 1.5m 이상 횡방향 이동
        labels.add("changing_lane")
        
        if lateral_displacement > 0:
            labels.add("changing_lane_to_left")
        else:
            labels.add("changing_lane_to_right")
    
    return labels
```

**임계값**:
- 횡방향 변위: `1.5 m`
- 분석 구간: 과거 10프레임 + 미래 10프레임

**신뢰도**: 0.80

## 3단계: Interaction-based Classification

상호작용 기반 분류는 다른 차량이나 보행자와의 관계를 분석합니다.

### 선행차 추종

Ego 차량 앞에 있는 선행차를 감지하고 추종 상태를 판단합니다:

```python
def _get_lead_vehicle(self, ego: EgoState, agents: List[TrackedObject]) -> Optional[TrackedObject]:
    ego_pos = np.array([ego.rear_axle.x, ego.rear_axle.y])
    ego_heading = ego.rear_axle.heading
    ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    
    lead_vehicle = None
    min_distance = float('inf')
    
    for agent in agents:
        if agent.tracked_object_type != TrackedObjectType.VEHICLE:
            continue
        
        agent_pos = np.array([agent.center.x, agent.center.y])
        relative_pos = agent_pos - ego_pos
        distance = np.linalg.norm(relative_pos)
        
        # 전방 거리 계산
        forward_distance = np.dot(relative_pos, ego_forward)
        
        if forward_distance > 0 and distance < LEAD_VEHICLE_DISTANCE:
            # 횡방향 거리 계산
            lateral_distance = abs(np.cross(relative_pos, ego_forward))
            
            # 같은 차선에 있는지 확인 (횡방향 2m 이내)
            if lateral_distance < 2.0 and distance < min_distance:
                min_distance = distance
                lead_vehicle = agent
    
    return lead_vehicle
```

```python
def _classify_interactions(self, window: ScenarioWindow) -> Set[str]:
    labels = set()
    
    ego = window.ego_current
    agents = window.agents_current
    
    # 선행차 감지
    lead_vehicle = self._get_lead_vehicle(ego, agents)
    
    if lead_vehicle:
        labels.add("following_lane_with_lead")
        
        # 느린 선행차 판단
        ego_speed = self._get_speed(ego)
        lead_speed = self._get_agent_speed(lead_vehicle)
        
        if lead_speed < ego_speed - SLOW_LEAD_SPEED_DIFF:
            labels.add("following_lane_with_slow_lead")
        
        # 특수 차량 판단
        if self._is_long_vehicle(lead_vehicle):
            labels.add("behind_long_vehicle")
        elif lead_vehicle.tracked_object_type == TrackedObjectType.BICYCLE:
            labels.add("behind_bike")
    else:
        labels.add("following_lane_without_lead")
    
    # 다중 객체 판단
    vehicles = [a for a in agents if a.tracked_object_type == TrackedObjectType.VEHICLE]
    pedestrians = [a for a in agents if a.tracked_object_type == TrackedObjectType.PEDESTRIAN]
    
    if len(vehicles) > MULTIPLE_VEHICLES_THRESHOLD:
        labels.add("near_multiple_vehicles")
    
    if len(pedestrians) > MULTIPLE_PEDESTRIANS_THRESHOLD:
        labels.add("near_multiple_pedestrians")
    
    return labels
```

**임계값**:
- `LEAD_VEHICLE_DISTANCE = 20.0 m`
- `SLOW_LEAD_SPEED_DIFF = 2.0 m/s`
- `MULTIPLE_VEHICLES_THRESHOLD = 10`
- `MULTIPLE_PEDESTRIANS_THRESHOLD = 3`
- 횡방향 거리: `2.0 m` (같은 차선 판단)

**신뢰도**:
- `following_lane_with_lead`: 0.85
- `following_lane_with_slow_lead`: 0.80
- `following_lane_without_lead`: 0.90
- `behind_long_vehicle`: 0.85
- `behind_bike`: 0.85
- `near_multiple_vehicles`: 0.95
- `near_multiple_pedestrians`: 0.95

## 4단계: Dynamics-based Classification

동역학 분류는 가속도 변화율(jerk)과 측면 가속도를 분석합니다.

### Jerk 계산

가속도의 시간 변화율을 계산하여 급가속/급감속을 감지합니다:

```python
def _compute_jerk(self, ego_states: List[EgoState]) -> float:
    if len(ego_states) < 3:
        return 0.0
    
    accels = []
    for state in ego_states:
        ax = state.dynamic_car_state.rear_axle_acceleration_2d.x
        ay = state.dynamic_car_state.rear_axle_acceleration_2d.y
        accel_mag = math.sqrt(ax**2 + ay**2)
        accels.append(accel_mag)
    
    # Jerk = 가속도 변화율
    jerk = (accels[-1] - accels[0]) / ((len(accels) - 1) * self.dt)
    return jerk
```

```python
def _classify_dynamics(self, window: ScenarioWindow) -> Set[str]:
    labels = set()
    
    if len(window.ego_history) >= 2:
        recent_states = window.ego_history[-2:] + [window.ego_current]
        jerk = self._compute_jerk(recent_states)
        
        if abs(jerk) > JERK_THRESHOLD:
            labels.add("high_magnitude_jerk")
    
    return labels
```

**임계값**:
- `JERK_THRESHOLD = 10.0 m/s³`
- 분석 구간: 최근 3 프레임 (0.15초, 20Hz 기준)

**신뢰도**: 0.95

### 측면 가속도 계산

회전 시 발생하는 측면 가속도를 계산합니다:

```python
def _compute_lateral_acceleration(self, ego_states: List[EgoState]) -> float:
    if len(ego_states) < 2:
        return 0.0
    
    state = ego_states[-1]
    speed = self._get_speed(state)
    
    if len(ego_states) >= 2:
        # 헤딩 변화율
        heading_rate = (ego_states[-1].rear_axle.heading - 
                       ego_states[-2].rear_axle.heading) / self.dt
        # 측면 가속도 = 속도 × 헤딩 변화율
        lateral_acc = speed * heading_rate
        return lateral_acc
    
    return 0.0
```

```python
if len(window.ego_history) >= 1:
    recent_states = window.ego_history[-1:] + [window.ego_current]
    lateral_acc = self._compute_lateral_acceleration(recent_states)
    
    if abs(lateral_acc) > LATERAL_ACCELERATION_THRESHOLD:
        labels.add("high_lateral_acceleration")
```

**임계값**:
- `LATERAL_ACCELERATION_THRESHOLD = 2.5 m/s²`
- 분석 구간: 최근 2 프레임 (0.1초, 20Hz 기준)

**신뢰도**: 0.95

## 라벨 통합 및 신뢰도 할당

4단계 분류가 완료되면 모든 라벨을 통합하고 신뢰도를 할당합니다:

```python
def classify(self, window: ScenarioWindow) -> ScenarioWindow:
    labels = set()
    
    # 4단계 분류 실행
    labels.update(self._classify_explicit_states(window))
    labels.update(self._classify_behaviors(window))
    labels.update(self._classify_interactions(window))
    labels.update(self._classify_dynamics(window))
    
    # ScenarioLabel 객체로 변환
    window.labels = [
        ScenarioLabel(
            label=label,
            confidence=self.confidence_levels.get(label, 0.5),
            category=self._get_label_category(label)
        )
        for label in sorted(labels)
    ]
    
    return window
```

### 신뢰도 레벨

라벨 타입에 따라 기본 신뢰도가 다릅니다:

- **High confidence (0.95~0.99)**: State-based, Map-based, Dynamics
  - 속도, 정지, 맵 기반 라벨
  - 동역학 특성 (jerk, 측면 가속도)

- **Medium confidence (0.80~0.95)**: Behavior-based, Interaction-based
  - 궤적 기반 행동 (차선변경, 회전)
  - 상호작용 (선행차 추종, 근접)

### 라벨 카테고리

라벨은 다음과 같은 카테고리로 분류됩니다:

```python
label_categories = {
    "speed_profile": ["low_magnitude_speed", "medium_magnitude_speed", "high_magnitude_speed"],
    "stationary": ["stationary", "stationary_in_traffic"],
    "turning": ["starting_left_turn", "starting_right_turn", 
                "starting_high_speed_turn", "starting_low_speed_turn"],
    "lane_change": ["changing_lane", "changing_lane_to_left", "changing_lane_to_right"],
    "following": ["following_lane_with_lead", "following_lane_with_slow_lead", 
                  "following_lane_without_lead"],
    "proximity": ["near_high_speed_vehicle", "near_long_vehicle", "near_multiple_vehicles",
                 "near_construction_zone_sign", "near_trafficcone_on_driveable", 
                 "near_barrier_on_driveable", "behind_long_vehicle", "behind_bike", 
                 "near_multiple_pedestrians"],
    "dynamics": ["high_magnitude_jerk", "high_lateral_acceleration"],
}
```

## 전체 라벨 목록

### 속도 프로파일
- `low_magnitude_speed`: 저속 주행 (< 10 km/h)
- `medium_magnitude_speed`: 중속 주행 (10~40 km/h)
- `high_magnitude_speed`: 고속 주행 (> 40 km/h)

### 정지 상황
- `stationary`: 정지 상태
- `stationary_in_traffic`: 교통체증 정지

### 회전
- `starting_left_turn`: 좌회전 시작
- `starting_right_turn`: 우회전 시작
- `starting_high_speed_turn`: 고속 회전
- `starting_low_speed_turn`: 저속 회전

### 차선변경
- `changing_lane`: 차선변경 중
- `changing_lane_to_left`: 좌측 차선변경
- `changing_lane_to_right`: 우측 차선변경

### 추종
- `following_lane_with_lead`: 선행차 추종
- `following_lane_with_slow_lead`: 느린 선행차 추종
- `following_lane_without_lead`: 선행차 없음

### 근접
- `near_high_speed_vehicle`: 고속 차량 근접
- `near_long_vehicle`: 대형 차량 근접
- `near_multiple_vehicles`: 다중 차량 근접
- `near_construction_zone_sign`: 공사구역 표지판 근접
- `near_trafficcone_on_driveable`: 콘 근접
- `near_barrier_on_driveable`: 장벽 근접
- `behind_long_vehicle`: 대형 차량 뒤
- `behind_bike`: 자전거 뒤
- `near_multiple_pedestrians`: 다중 보행자 근접

### 동역학
- `high_magnitude_jerk`: 높은 가속도 변화율
- `high_lateral_acceleration`: 높은 측면 가속도

## 확장 가능성

새로운 라벨을 추가하려면:

1. **분류 메서드 추가/수정**: 해당 단계의 분류 메서드에 로직 추가
2. **신뢰도 설정**: `confidence_levels` 딕셔너리에 신뢰도 추가
3. **카테고리 분류**: `label_categories`에 적절한 카테고리 추가
4. **임계값 조정**: 필요시 클래스 상수에 임계값 추가

## 다음 단계

- [처리 파이프라인 문서](./08_pipeline.md) - 분류기가 파이프라인에서 어떻게 사용되는지
- [데이터 구조 문서](./02_data_structures.md) - ScenarioWindow 구조 상세

