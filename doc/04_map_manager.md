# 맵 관리자 (MapManager)

MapManager는 SQLite 기반 맵 데이터베이스를 관리하고 공간 쿼리를 제공하는 핵심 컴포넌트입니다. NuPlan 프레임워크와 호환되는 맵 데이터 구조를 제공합니다.

## 개요

MapManager는 다음과 같은 기능을 제공합니다:

- SQLite 맵 데이터베이스 로딩 및 관리
- 다양한 맵 레이어 지원 (Lane, LaneConnector, Roadblock, StopLine, Crosswalk 등)
- STRtree 기반 공간 인덱싱으로 효율적인 공간 쿼리
- NuPlan 프레임워크 호환 API 제공

## 맵 데이터 구조

### 지원하는 맵 레이어

시스템은 다음과 같은 맵 레이어를 지원합니다:

1. **Lane**: 일반 차선
2. **LaneConnector**: 차선 연결부 (교차로 등)
3. **Roadblock**: 도로 블록
4. **RoadblockConnector**: 도로 블록 연결부
5. **StopLine**: 정지선
6. **Crosswalk**: 횡단보도
7. **CarparkArea**: 주차장 영역
8. **Laneside**: 차선 측면
9. **RoadLight**: 차량용 신호등
10. **PedestrianLight**: 보행자용 신호등

### SQLite 데이터베이스 구조

맵 데이터는 SQLite 데이터베이스에 저장되며, 각 레이어는 별도의 테이블로 관리됩니다:

- `lanes`: 차선 정보
- `lane_connectors`: 차선 연결부 정보
- `roadblocks`: 도로 블록 정보
- `roadblock_connectors`: 도로 블록 연결부 정보
- `stop_lines`: 정지선 정보
- `crosswalks`: 횡단보도 정보
- `carpark_areas`: 주차장 영역 정보
- `lanesides`: 차선 측면 정보
- `roadlights`: 차량용 신호등 정보
- `pedestrian_lights`: 보행자용 신호등 정보
- `lane_successors`: 차선 후속 관계
- `roadblock_interior_edges`: 도로 블록 내부 차선 관계
- `rbc_interior_edges`: 도로 블록 연결부 내부 차선 연결부 관계

## 클래스 구조

### MapManager

```python
class MapManager:
    def __init__(self, sqlite_path: str):
        # SQLite 파일 경로로 초기화
    
    def initialize_all_layers(self) -> None:
        # 모든 레이어 로딩 및 인덱싱
    
    # 공간 쿼리 메서드
    def get_proximal_map_objects(self, point, radius: float, layers) -> Dict
    def get_one_map_object(self, point, layer, search_radius: float) -> Optional[object]
    def get_distance_to_nearest_map_object(self, point, layer, search_radius: float) -> Tuple
    def get_nearest_lane(self, point_xy: Tuple[float, float], search_radius: float) -> Tuple
    
    # 차선 관련 메서드
    def get_lane_successors(self, lane_id: Union[str, int]) -> List[str]
    def get_lane_predecessors(self, lane_id: Union[str, int]) -> List[str]
    def get_lane_polygon(self, lane_id: Union[str, int]) -> Optional[BaseGeometry]
    def get_lane_baseline(self, lane_id: Union[str, int], sampled: bool) -> Optional[BaseGeometry]
    def get_lane_speed_limit(self, lane_id: Union[str, int]) -> Optional[float]
    def get_lane_width(self, lane_id: Union[str, int]) -> Optional[float]
    def project_onto_lane(self, lane_id: Union[str, int], point_xy: Tuple[float, float], sampled: bool) -> Optional[Tuple]
    
    # 차선 방향 및 연결 정보
    def get_lane_direction_info(self, lane_id: Union[str, int]) -> Dict
    def get_adjacent_lanes(self, lane_id: Union[str, int]) -> Dict[str, Optional[str]]
    def is_lane_junction(self, lane_id: Union[str, int]) -> bool
    
    # 신호등 관련 메서드
    def get_traffic_light_for_lane(self, lane_id: Union[str, int]) -> Optional[RoadLight]
    def get_roadlight(self, roadlight_id: int) -> Optional[RoadLight]
    def get_roadlights_by_lane(self, lane_id: Union[str, int]) -> List[RoadLight]
    def get_pedestrian_light(self, pedestrian_light_id: int) -> Optional[PedestrianLight]
    def get_pedestrian_lights_by_crosswalk(self, crosswalk_id: int) -> List[PedestrianLight]
```

## 데이터 타입

### Lane

일반 차선을 나타내는 데이터 구조입니다.

```python
@dataclass(eq=False)
class Lane:
    id: str
    roadblock_id: Optional[str]
    polygon: BaseGeometry              # 차선 폴리곤
    baseline: BaseGeometry             # 차선 중심선
    baseline_sampled: Optional[BaseGeometry]  # 샘플링된 중심선
    left_boundary: BaseGeometry        # 좌측 경계
    right_boundary: BaseGeometry       # 우측 경계
    length_m: float                    # 차선 길이 (미터)
    speed_limit_mps: Optional[float]   # 제한 속도 (m/s)
    width_m: Optional[float]           # 차선 폭 (미터)
    baseline_path: Optional[Path]      # 중심선 경로
    
    # 방향 및 연결 정보
    is_bidirectional: Optional[bool]   # 양방향 여부
    start_node_id: Optional[int]       # 시작 노드 ID
    end_node_id: Optional[int]         # 종료 노드 ID
    junction: Optional[int]            # 교차로 ID
    lane_type: Optional[int]           # 차선 타입
    sub_type: Optional[int]            # 차선 서브 타입
    vehicle_traffic_light_id: Optional[int]  # 연결된 신호등 ID
    left_link_id: Optional[int]        # 좌측 인접 차선 ID
    right_link_id: Optional[int]       # 우측 인접 차선 ID
    
    # 그래프 엣지 (로딩 후 연결)
    outgoing_edges: List['Lane']       # 후속 차선들
    incoming_edges: List['Lane']       # 선행 차선들
```

### LaneConnector

차선 연결부를 나타내는 데이터 구조입니다. 주로 교차로에서 차선 간 연결을 나타냅니다.

```python
@dataclass(eq=False)
class LaneConnector:
    id: str
    roadblock_connector_id: Optional[str]
    polygon: BaseGeometry
    baseline: BaseGeometry
    length_m: float
    baseline_sampled: Optional[BaseGeometry]
    from_lane_id: Optional[str]        # 출발 차선 ID
    to_lane_id: Optional[str]          # 도착 차선 ID
    speed_limit_mps: Optional[float]
    width_m: Optional[float]
    left_boundary: Optional[BaseGeometry]
    right_boundary: Optional[BaseGeometry]
    baseline_path: Optional[Path]
```

### Roadblock

도로 블록을 나타내는 데이터 구조입니다. 여러 차선을 포함하는 도로 구간입니다.

```python
@dataclass
class Roadblock:
    id: str
    polygon: BaseGeometry
    interior_edges: List[str]          # 내부 차선 ID 리스트
    interior_edge_objs: List['Lane']   # 내부 차선 객체 리스트 (로딩 후)
    incoming_edges: List['Roadblock']  # 진입 도로 블록들
    outgoing_edges: List['Roadblock']  # 진출 도로 블록들
```

### StopLine

정지선을 나타내는 데이터 구조입니다.

```python
@dataclass
class StopLine:
    id: str
    linestring: BaseGeometry           # 정지선 라인스트링
```

### Crosswalk

횡단보도를 나타내는 데이터 구조입니다.

```python
@dataclass
class Crosswalk:
    id: str
    polygon: BaseGeometry              # 횡단보도 폴리곤
```

### RoadLight

차량용 신호등을 나타내는 데이터 구조입니다.

```python
@dataclass
class RoadLight:
    id: int
    lane_id: int                       # 연결된 차선 ID
    light_type: int                    # 신호등 타입
    sub_type: int                      # 서브 타입
    div: int                           # 구분
    uturn: int                         # U턴 가능 여부
    num_stop_lines: int                # 연결된 정지선 수
    stop_line_ids: List[int]           # 연결된 정지선 ID 리스트
    num_points: int                    # 포인트 수
    linestring: BaseGeometry           # 신호등 위치 라인스트링
```

### PedestrianLight

보행자용 신호등을 나타내는 데이터 구조입니다.

```python
@dataclass
class PedestrianLight:
    id: int
    num_crosswalks: int                # 연결된 횡단보도 수
    crosswalk_ids: List[int]           # 연결된 횡단보도 ID 리스트
    light_type: int                    # 신호등 타입
    direction: float                   # 방향
    x: float                           # X 좌표
    y: float                           # Y 좌표
```

## 초기화 과정

### 초기화 흐름도

```mermaid
flowchart TD
    A[MapManager 초기화] --> B[SQLite 파일 연결]
    B --> C[레이어 로딩]
    C --> C1[_load_lanes]
    C --> C2[_load_lane_connectors]
    C --> C3[_load_roadblocks]
    C --> C4[_load_stop_lines]
    C --> C5[_load_crosswalks]
    C --> C6[_load_roadlights]
    C --> C7[_load_pedestrian_lights]
    C --> C8[_load_lane_graph]
    
    C1 --> D[그래프 연결]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    C6 --> D
    C7 --> D
    C8 --> D
    
    D --> E[_wire_graphs<br/>차선 후속/선행 관계 연결]
    E --> F[STRtree 인덱스 구축]
    F --> F1[차선 인덱스]
    F --> F2[차선 연결부 인덱스]
    F --> F3[도로 블록 인덱스]
    F --> F4[정지선 인덱스]
    F --> F5[횡단보도 인덱스]
    F --> F6[신호등 인덱스]
    
    F1 --> G[초기화 완료]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
    F6 --> G
```

### 1. 레이어 로딩

```python
def initialize_all_layers(self) -> None:
    conn = sqlite3.connect(self.sqlite_path)
    try:
        self._load_lanes(conn)
        self._load_lane_connectors(conn)
        self._load_roadblocks(conn)
        self._load_roadblock_connectors(conn)
        self._load_stop_lines(conn)
        self._load_crosswalks(conn)
        self._load_carpark_areas(conn)
        self._load_lanesides(conn)
        self._load_roadlights(conn)
        self._load_pedestrian_lights(conn)
        self._load_lane_graph(conn)
    finally:
        conn.close()
    
    # 객체 그래프 연결
    self._wire_graphs()
    
    # STRtree 인덱스 구축
    self._build_spatial_indexes()
```

### 2. 공간 인덱싱 (STRtree)

효율적인 공간 쿼리를 위해 Shapely의 STRtree를 사용합니다:

```python
# 차선 인덱스 예시
self._lane_geoms = []      # 기하 객체 리스트
self._lane_ids = []        # ID 리스트
self._lane_gid_map = {}    # 기하 객체 ID → 차선 ID 매핑

for lid, lane in self._lanes.items():
    g = lane.polygon
    if g is not None and not g.is_empty:
        self._lane_geoms.append(g)
        self._lane_ids.append(lid)
        self._lane_gid_map[id(g)] = lid

self._lane_tree = STRtree(self._lane_geoms)
```

STRtree는 R-tree의 변형으로, 공간 쿼리 성능을 크게 향상시킵니다.

### 3. 그래프 연결

차선 간 연결 관계를 객체 레벨에서 연결합니다:

```python
def _wire_graphs(self) -> None:
    # 차선 후속/선행 관계 연결
    for lid, lane in self._lanes.items():
        lane.outgoing_edges = [self._lanes[sid] 
                              for sid in self._lane_successors.get(lid, []) 
                              if sid in self._lanes]
        lane.incoming_edges = [self._lanes[pid] 
                              for pid in self._lane_predecessors.get(lid, []) 
                              if pid in self._lanes]
    
    # 도로 블록 내부 차선 연결
    for rb_id, rb in self._roadblocks.items():
        rb.interior_edge_objs = [self._lanes[lid] 
                                for lid in rb.interior_edges 
                                if lid in self._lanes]
```

## 맵 데이터 구조 관계

```mermaid
graph TD
    A[MapManager] --> B[Lane]
    A --> C[LaneConnector]
    A --> D[Roadblock]
    A --> E[StopLine]
    A --> F[Crosswalk]
    A --> G[RoadLight]
    A --> H[PedestrianLight]
    
    B --> B1[후속 차선<br/>outgoing_edges]
    B --> B2[선행 차선<br/>incoming_edges]
    B --> B3[좌측 인접 차선<br/>left_link_id]
    B --> B4[우측 인접 차선<br/>right_link_id]
    
    D --> D1[내부 차선<br/>interior_edges]
    D --> D2[진입 도로 블록<br/>incoming_edges]
    D --> D3[진출 도로 블록<br/>outgoing_edges]
    
    C --> C1[출발 차선<br/>from_lane_id]
    C --> C2[도착 차선<br/>to_lane_id]
    
    G --> G1[차선 ID<br/>lane_id]
    G --> G2[정지선 ID<br/>stop_line_ids]
    
    H --> H1[횡단보도 ID<br/>crosswalk_ids]
    
    B -.->|차선 그래프| B
    D -.->|도로 블록 그래프| D
```

## 공간 쿼리 API

### get_proximal_map_objects

주어진 점 주변의 맵 객체들을 반환합니다.

```python
def get_proximal_map_objects(
    self, 
    point, 
    radius: float, 
    layers: Iterable[LayerLike]
) -> Dict[LayerLike, List[object]]:
    """
    입력:
        point: 쿼리 포인트 (x, y) 또는 Point 객체
        radius: 검색 반경 (미터)
        layers: 검색할 레이어 리스트
    
    반환:
        레이어별 맵 객체 리스트 딕셔너리
    """
```

**사용 예시**:
```python
query_point = Point2D(230388.61912, 424695.37128)
radius = 30.0
layers = [SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK]

results = map_manager.get_proximal_map_objects(query_point, radius, layers)
lanes = results[SemanticMapLayer.LANE]
crosswalks = results[SemanticMapLayer.CROSSWALK]
```

### get_nearest_lane

가장 가까운 차선을 찾습니다.

```python
def get_nearest_lane(
    self, 
    point_xy: Tuple[float, float], 
    search_radius: float = 50.0
) -> Tuple[Optional[str], float]:
    """
    입력:
        point_xy: 쿼리 포인트 (x, y)
        search_radius: 검색 반경 (미터)
    
    반환:
        (차선 ID, 거리) 튜플
    """
```

**사용 예시**:
```python
point = (230388.61912, 424695.37128)
lane_id, distance = map_manager.get_nearest_lane(point, search_radius=50.0)
if lane_id:
    print(f"가장 가까운 차선: {lane_id}, 거리: {distance:.2f}m")
```

### project_onto_lane

점을 차선 중심선에 투영합니다.

```python
def project_onto_lane(
    self, 
    lane_id: Union[str, int], 
    point_xy: Tuple[float, float], 
    sampled: bool = True
) -> Optional[Tuple[float, Tuple[float, float], float]]:
    """
    입력:
        lane_id: 차선 ID
        point_xy: 투영할 점 (x, y)
        sampled: 샘플링된 중심선 사용 여부
    
    반환:
        (중심선상 거리, 투영점, 유클리드 거리) 튜플
    """
```

**사용 예시**:
```python
lane_id = "12345"
point = (230388.61912, 424695.37128)
result = map_manager.project_onto_lane(lane_id, point)
if result:
    s, foot_point, distance = result
    print(f"중심선상 거리: {s:.2f}m, 투영점: {foot_point}, 거리: {distance:.2f}m")
```

## 차선 그래프 API

### get_lane_successors / get_lane_predecessors

차선의 후속/선행 차선을 반환합니다.

```python
successors = map_manager.get_lane_successors("12345")
predecessors = map_manager.get_lane_predecessors("12345")
```

### get_adjacent_lanes

인접한 좌측/우측 차선을 반환합니다.

```python
adjacent = map_manager.get_adjacent_lanes("12345")
left_lane_id = adjacent['left']
right_lane_id = adjacent['right']
```

### is_lane_junction

차선이 교차로에 속하는지 확인합니다.

```python
is_junction = map_manager.is_lane_junction("12345")
```

## 신호등 API

### get_traffic_light_for_lane

차선에 연결된 신호등을 반환합니다.

```python
roadlight = map_manager.get_traffic_light_for_lane("12345")
if roadlight:
    print(f"신호등 ID: {roadlight.id}, 타입: {roadlight.light_type}")
```

### get_roadlights_by_lane

특정 차선에 연결된 모든 신호등을 반환합니다.

```python
roadlights = map_manager.get_roadlights_by_lane("12345")
```

### get_pedestrian_lights_by_crosswalk

횡단보도에 연결된 보행자 신호등을 반환합니다.

```python
pedestrian_lights = map_manager.get_pedestrian_lights_by_crosswalk(123)
```

## 좌표계

맵 데이터는 UTM (Universal Transverse Mercator) 좌표계를 사용합니다. 맵 원점은 `DefaultParams.py`에서 설정됩니다:

```python
MAP_ORIGIN_X = 230388.61912  # UTM X 좌표 (미터)
MAP_ORIGIN_Y = 424695.37128  # UTM Y 좌표 (미터)
```

## 성능 최적화

### STRtree 인덱싱

공간 쿼리 성능을 위해 STRtree를 사용합니다:

- **쿼리 시간**: O(log n) (n은 객체 수)
- **메모리**: O(n)
- **구축 시간**: O(n log n)

### 지연 로딩

필요한 레이어만 로딩할 수 있도록 설계되었지만, 현재는 모든 레이어를 한 번에 로딩합니다.

### 캐싱

맵 객체는 메모리에 캐시되어 반복 쿼리 시 빠르게 접근할 수 있습니다.

## 사용 예시

### 기본 사용

```python
from MapManager import MapManager

# 맵 매니저 초기화
map_manager = MapManager("src/map.sqlite")
map_manager.initialize_all_layers()

# 근접 차선 검색
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

query_point = Point2D(230388.61912, 424695.37128)
radius = 30.0

results = map_manager.get_proximal_map_objects(
    query_point, 
    radius, 
    [SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK]
)

lanes = results[SemanticMapLayer.LANE]
for lane in lanes:
    print(f"차선 ID: {lane.id}, 길이: {lane.length_m}m")
```

### 차선 그래프 탐색

```python
# 시작 차선
start_lane_id = "12345"

# 후속 차선 탐색
current_lane_id = start_lane_id
for i in range(5):  # 5단계까지 탐색
    successors = map_manager.get_lane_successors(current_lane_id)
    if not successors:
        break
    current_lane_id = successors[0]
    print(f"단계 {i+1}: {current_lane_id}")
```

## 다음 단계

- [시각화 시스템 문서](./06_visualization.md) - 맵 데이터를 시각화에 사용
- [JSON 로그 로더 문서](./05_json_loader.md) - 맵 데이터를 로그 파싱에 사용

