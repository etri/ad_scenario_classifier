# 시각화 시스템 (CustomScenarioVisualizer)

CustomScenarioVisualizer는 라벨링된 시나리오를 시각화하여 PNG 이미지로 저장하는 컴포넌트입니다. Ego 중심 좌표계를 사용하여 직관적인 시각화를 제공합니다.

## 개요

CustomScenarioVisualizer는 다음과 같은 요소를 시각화합니다:

- **맵 요소**: 차선, 횡단보도
- **Ego 차량**: 현재 위치 및 방향
- **주변 객체**: 차량, 보행자, 자전거 등
- **궤적**: 과거 및 미래 궤적
- **라벨 정보**: 분류된 라벨 오버레이

## 클래스 구조

```python
class CustomScenarioVisualizer:
    def __init__(self, map_manager=None):
        # 맵 매니저 및 시각화 설정 초기화
    
    def visualize(self, window: ScenarioWindow, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        # 시나리오 시각화 및 이미지 생성
    
    # 렌더링 메서드들
    def _setup_coordinate_system(self, ego_state: EgoState)
    def _transform_coordinates(self, points)
    def _render_map(self, ax)
    def _render_ego_vehicle(self, ax, ego_state: EgoState)
    def _render_agents(self, ax, agents: List[TrackedObject])
    def _render_trajectories(self, ax, window: ScenarioWindow)
    def _render_scenario_labels(self, ax, labels: List[ScenarioLabel])
    def _configure_axes(self, ax)
    def _convert_to_image(self, fig) -> Optional[np.ndarray]
    def _save_image(self, img: np.ndarray, save_path: str)
```

## Ego 중심 좌표계

시각화는 Ego 차량을 중심으로 한 좌표계를 사용합니다. 이를 통해 모든 객체의 상대적 위치를 직관적으로 파악할 수 있습니다.

### 좌표계 변환

```python
def _setup_coordinate_system(self, ego_state: EgoState):
    """Ego 중심 좌표계 설정"""
    self.origin = ego_state.rear_axle.array  # [x, y, heading]
    self.angle = ego_state.rear_axle.heading
    
    # 회전 행렬 생성
    self.rot_mat = np.array([
        [np.cos(self.angle), -np.sin(self.angle)],
        [np.sin(self.angle), np.cos(self.angle)]
    ], dtype=np.float64)

def _transform_coordinates(self, points):
    """월드 좌표를 Ego 중심 좌표로 변환"""
    if points.ndim == 1:
        points = points.reshape(1, -1)
    # 평행이동 후 회전
    return np.matmul(points - self.origin[:2], self.rot_mat)
```

### 변환 과정

1. **평행이동**: Ego 위치를 원점으로 이동
2. **회전**: Ego 헤딩을 0도로 회전

변환 후:
- Ego 차량은 항상 원점 (0, 0)에 위치
- Ego 헤딩은 항상 0도 (동쪽 방향)
- 전방은 +X 방향, 좌측은 +Y 방향

## 시각화 설정

### 기본 설정

```python
self.bounds = 60        # 시각화 범위 (미터)
self.offset = 20        # X축 오프셋 (전방 시야 확보)
self.figsize = (10, 10) # 이미지 크기 (인치)
```

### 색상 매핑

```python
self.agent_colors = {
    TrackedObjectType.VEHICLE: "#001eff",    # 파란색
    TrackedObjectType.PEDESTRIAN: "#9500ff", # 보라색
    TrackedObjectType.BICYCLE: "#ff0059",    # 분홍색
    TrackedObjectType.EGO: "#ff7f0e"         # 주황색
}
```

## 렌더링 순서

시각화는 다음 순서로 렌더링됩니다 (배경 → 전경):

1. **맵 요소** (zorder=0~3)
2. **궤적** (zorder=5~6)
3. **주변 객체** (zorder=4)
4. **Ego 차량** (zorder=10~11)
5. **라벨 정보** (zorder=1000)

## 맵 렌더링

### 차선 렌더링

```python
def _render_map(self, ax):
    # 근접 차선 검색
    query_point = Point2D(self.origin[0], self.origin[1])
    road_elements = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    
    road_objects = self.map_manager.get_proximal_map_objects(
        query_point, self.bounds + self.offset, road_elements
    )
    
    all_lanes = (road_objects[SemanticMapLayer.LANE] + 
                road_objects[SemanticMapLayer.LANE_CONNECTOR])
    
    for lane in all_lanes:
        # 차선 폴리곤
        polygon_coords = np.array(lane.polygon.exterior.xy).T
        transformed_coords = self._transform_coordinates(polygon_coords)
        
        lane_patch = patches.Polygon(
            transformed_coords, 
            color="lightgray", 
            alpha=0.4, 
            ec=None, 
            zorder=0
        )
        ax.add_patch(lane_patch)
        
        # 차선 중심선
        centerline = np.array([[s.x, s.y] for s in lane.baseline_path.discrete_path])
        transformed_centerline = self._transform_coordinates(centerline)
        
        ax.plot(
            transformed_centerline[:, 0], 
            transformed_centerline[:, 1],
            color="gray", 
            alpha=0.5, 
            linestyle="--", 
            linewidth=1, 
            zorder=1
        )
```

### 횡단보도 렌더링

```python
# 횡단보도 검색
crosswalks = self.map_manager.get_proximal_map_objects(
    query_point, self.bounds + self.offset, [SemanticMapLayer.CROSSWALK]
)

for crosswalk in crosswalks[SemanticMapLayer.CROSSWALK]:
    polygon_coords = np.array(crosswalk.polygon.exterior.coords.xy).T
    transformed_coords = self._transform_coordinates(polygon_coords)
    
    crosswalk_patch = patches.Polygon(
        transformed_coords,
        color="gray",
        alpha=0.4,
        ec=None,
        zorder=3,
        hatch="///"  # 사선 패턴
    )
    ax.add_patch(crosswalk_patch)
```

## Ego 차량 렌더링

```python
def _render_ego_vehicle(self, ax, ego_state: EgoState):
    # 차량 외곽선
    footprint_coords = np.array(ego_state.car_footprint.geometry.exterior.xy).T
    transformed_footprint = self._transform_coordinates(footprint_coords)
    
    ego_patch = patches.Polygon(
        transformed_footprint,
        ec=self.agent_colors[TrackedObjectType.EGO],
        fill=False,
        linewidth=2,
        zorder=10
    )
    ax.add_patch(ego_patch)
    
    # 방향 표시 (전방)
    length_indicator = self.ego_params.length * 0.75
    ax.plot(
        [1.69, 1.69 + length_indicator],  # 후축에서 전방으로
        [0, 0],
        color=self.agent_colors[TrackedObjectType.EGO],
        linewidth=2,
        zorder=11
    )
```

Ego 차량은 항상 원점 (0, 0)에 위치하며, 전방이 +X 방향입니다.

## 주변 객체 렌더링

```python
def _render_agents(self, ax, agents: List[TrackedObject]):
    for agent in agents:
        # 위치 변환
        center = self._transform_coordinates(agent.center.array.reshape(1, -1))[0]
        angle = agent.center.heading - self.angle  # 상대 각도
        
        # 바운딩 박스
        agent_coords = np.array(agent.box.geometry.exterior.xy).T
        transformed_coords = self._transform_coordinates(agent_coords)
        
        color = self.agent_colors.get(agent.tracked_object_type, "black")
        
        agent_patch = patches.Polygon(
            transformed_coords,
            ec=color,
            fill=False,
            alpha=1.0,
            linewidth=1.5,
            zorder=4
        )
        ax.add_patch(agent_patch)
        
        # 방향 화살표 (움직이는 객체만)
        if hasattr(agent, 'velocity') and agent.velocity is not None:
            velocity_magnitude = np.linalg.norm(agent.velocity.array)
            if velocity_magnitude > 0.3:  # 0.3 m/s 이상만 표시
                direction = np.array([np.cos(angle), np.sin(angle)]) * agent.box.length / 2
                arrow_start = center
                arrow_end = center + direction
                
                ax.plot(
                    [arrow_start[0], arrow_end[0]], 
                    [arrow_start[1], arrow_end[1]],
                    color=color, 
                    linewidth=1, 
                    zorder=4
                )
```

## 궤적 렌더링

### Ego 궤적

```python
def _render_trajectories(self, ax, window: ScenarioWindow):
    # 과거 궤적 (실선)
    if window.ego_history:
        history_points = np.array([state.rear_axle.array for state in window.ego_history])
        transformed_history = self._transform_coordinates(history_points)
        
        ax.plot(
            transformed_history[:, 0], 
            transformed_history[:, 1],
            color=self.agent_colors[TrackedObjectType.EGO],
            alpha=0.5,
            linewidth=2,
            zorder=6,
            label='Ego History'
        )
    
    # 미래 궤적 (점선)
    if window.ego_future:
        future_points = np.array([state.rear_axle.array for state in window.ego_future])
        transformed_future = self._transform_coordinates(future_points)
        
        ax.plot(
            transformed_future[:, 0], 
            transformed_future[:, 1],
            color="blue",
            alpha=0.5,
            linewidth=2,
            zorder=6,
            linestyle=":",
            label='Ego Future'
        )
```

### 주변 객체 궤적

```python
# 에이전트 궤적 추출
agent_trajectories = self._extract_agent_trajectories(window)

for agent_id, trajectory_data in agent_trajectories.items():
    agent_type = trajectory_data.get('type', TrackedObjectType.VEHICLE)
    history_positions = trajectory_data.get('history', [])
    future_positions = trajectory_data.get('future', [])
    
    base_color = self.agent_colors.get(agent_type, "gray")
    
    # 과거 궤적 (실선)
    if len(history_positions) >= 3:
        history_array = np.array(history_positions)
        transformed_agent_history = self._transform_coordinates(history_array)
        
        ax.plot(
            transformed_agent_history[:, 0],
            transformed_agent_history[:, 1],
            color=base_color,
            alpha=0.3,
            linewidth=1,
            zorder=5,
            linestyle='-'
        )
    
    # 미래 궤적 (점선)
    if len(future_positions) >= 3:
        future_array = np.array(future_positions)
        transformed_agent_future = self._transform_coordinates(future_array)
        
        ax.plot(
            transformed_agent_future[:, 0],
            transformed_agent_future[:, 1],
            color=base_color,
            alpha=0.25,
            linewidth=1,
            zorder=5,
            linestyle='--'
        )
```

### 궤적 추출

```python
def _extract_agent_trajectories(self, window: ScenarioWindow) -> Dict[str, Dict]:
    agent_trajectories = {}
    
    # 현재 프레임의 에이전트 수집
    for agent in window.agents_current:
        agent_id = agent.track_token
        agent_trajectories[agent_id] = {
            'type': agent.tracked_object_type,
            'current_position': np.array([agent.center.x, agent.center.y]),
            'history': [],
            'future': []
        }
    
    # 과거 위치 수집
    for time_step, agents_at_time in enumerate(window.agents_history):
        for agent in agents_at_time:
            agent_id = agent.track_token
            if agent_id in agent_trajectories:
                position = np.array([agent.center.x, agent.center.y])
                agent_trajectories[agent_id]['history'].append(position)
    
    # 미래 위치 수집
    for time_step, agents_at_time in enumerate(window.agents_future):
        for agent in agents_at_time:
            agent_id = agent.track_token
            if agent_id in agent_trajectories:
                position = np.array([agent.center.x, agent.center.y])
                agent_trajectories[agent_id]['future'].append(position)
    
    return agent_trajectories
```

## 라벨 정보 렌더링

```python
def _render_scenario_labels(self, ax, labels: List[ScenarioLabel]):
    if not labels:
        return
    
    # 카테고리별 그룹화
    categories = {}
    for label in labels:
        cat = label.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(label)
    
    # 텍스트 생성
    text_lines = []
    for category, cat_labels in sorted(categories.items()):
        text_lines.append(f"{category.upper()}:")
        for label in cat_labels:
            text_lines.append(f"  • {label.label} ({label.confidence:.2f})")
        text_lines.append("")  # 빈 줄
    
    full_text = "\n".join(text_lines).strip()
    
    # 텍스트 오버레이
    if full_text:
        ax.text(
            0.02, 0.98,  # 좌상단
            full_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            ),
            zorder=1000
        )
```

라벨은 카테고리별로 그룹화되어 표시되며, 각 라벨의 신뢰도도 함께 표시됩니다.

## 이미지 생성 및 저장

### 좌표축 설정

```python
def _configure_axes(self, ax):
    ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
    ax.set_ylim(ymin=-self.bounds, ymax=self.bounds)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")  # 축 숨김
    plt.tight_layout(pad=0)
```

### 이미지 변환

```python
def _convert_to_image(self, fig) -> Optional[np.ndarray]:
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    
    # Matplotlib 버전에 따라 다른 방법 사용
    try:
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf).reshape(int(height), int(width), 4)
        img = img[:, :, :3]  # RGBA to RGB
    except AttributeError:
        # 구버전 호환
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            int(height), int(width), 3
        )
    
    return img
```

### 이미지 저장

```python
def _save_image(self, img: np.ndarray, save_path: str):
    import cv2
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # RGB to BGR 변환 (OpenCV는 BGR 사용)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(save_path, img_bgr)
    
    if not success:
        print(f"Warning: cv2.imwrite failed for {save_path}")
```

## 사용 예시

### 기본 사용

```python
from CustomScenarioVisualizer import CustomScenarioVisualizer
from MapManager import MapManager

# 맵 매니저 초기화
map_manager = MapManager("src/map.sqlite")
map_manager.initialize_all_layers()

# 시각화 생성
visualizer = CustomScenarioVisualizer(map_manager=map_manager)

# 시나리오 시각화
save_path = "./imgs/scenario_000100.png"
img = visualizer.visualize(window, save_path=save_path)
```

### 배치 시각화

```python
for window in windows:
    viz_path = os.path.join(output_dir, f"scenario_{window.center_idx:06d}.png")
    img = visualizer.visualize(window, save_path=viz_path)
    
    if img is not None:
        success_count += 1
```

## 시각화 범위

- **X축**: -40m ~ +80m (전방 80m, 후방 40m)
- **Y축**: -60m ~ +60m (좌우 각 60m)
- **총 범위**: 120m × 120m

Ego 차량은 항상 원점 (0, 0)에 위치하며, 전방이 +X 방향입니다.

## 에러 처리

시각화 중 에러가 발생해도 다른 시나리오 처리를 계속할 수 있도록 에러를 처리합니다:

```python
try:
    # 시각화 수행
    ...
except Exception as e:
    print(f"Warning: Custom visualization failed for scenario {window.center_idx}")
    print(f"   Error: {e}")
    return None
```

## 성능 고려사항

### 렌더링 시간

- 맵 쿼리: O(log n) (STRtree 사용)
- 좌표 변환: O(m) (m은 점의 수)
- 이미지 생성: O(width × height)

### 메모리 사용

- 이미지 크기: 약 10MB (1000×1000 픽셀, RGB)
- Matplotlib figure: 추가 메모리 사용

### 최적화

- 불필요한 객체 필터링 (범위 밖 객체 제외)
- 궤적 길이 제한
- 배치 처리 시 메모리 해제

## 다음 단계

- [처리 파이프라인 문서](./08_pipeline.md) - 시각화가 파이프라인에서 사용되는 방법
- [데이터 출력 문서](./07_export.md) - 시각화와 함께 JSON 출력

