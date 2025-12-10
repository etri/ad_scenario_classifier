# 처리 파이프라인

이 문서는 시나리오 분류기 시스템의 전체 처리 파이프라인에 대한 상세 설명을 제공합니다.

## 개요

처리 파이프라인은 JSON 로그 파일을 입력받아 라벨링된 시나리오와 시각화 이미지를 생성하는 전체 과정을 포함합니다.

## 전체 파이프라인 흐름

```mermaid
flowchart TD
    A[1. 초기화] --> B[맵 매니저 초기화]
    A --> C[설정 로딩]
    B --> D[2. 로그 파일 로딩]
    C --> D
    D --> E[JSON 파일 읽기]
    D --> F[로그 엔트리 파싱]
    D --> G[데이터 추출]
    E --> H[3. 윈도우 추출]
    F --> H
    G --> H
    H --> I[101-epoch 윈도우 생성]
    H --> J[샘플링 step_size]
    H --> K[윈도우 리스트 생성]
    I --> L[4. 시나리오 분류]
    J --> L
    K --> L
    L --> M[각 윈도우에 대해 분류 수행]
    L --> N[라벨 할당]
    M --> O[5. JSON 출력]
    N --> O
    O --> P[각 시나리오를 JSON 파일로 저장]
    O --> Q[요약 파일 생성]
    P --> R[6. 시각화 선택적]
    Q --> R
    R --> S[각 시나리오 시각화]
    R --> T[PNG 이미지 저장]
    S --> U[7. 통계 생성]
    T --> U
    U --> V[라벨 분포 및 통계 출력]
```

## process_log_file 함수

전체 파이프라인은 `process_log_file` 함수로 구현되어 있습니다.

### 함수 시그니처

```python
def process_log_file(
    log_file_path: str, 
    map_manager=None,
    output_dir_json: str = "./labeled_scenarios",
    output_dir_viz: str = "./imgs",
    visualize: bool = True,
    max_scenarios: int = -1,
    step_size: int = 1
):
```

### 파라미터

- **log_file_path** (str): JSON 로그 파일 경로
- **map_manager**: 맵 매니저 객체 (None이면 자동 생성)
- **output_dir_json** (str): JSON 출력 디렉토리
- **output_dir_viz** (str): 시각화 이미지 출력 디렉토리
- **visualize** (bool): 시각화 생성 여부
- **max_scenarios** (int): 처리할 최대 시나리오 수 (-1 = 전체)
- **step_size** (int): 시나리오 생성 스텝 크기 (1 = 전체, 큰 값 = 샘플링)

## 단계별 상세 설명

### 1단계: 초기화

```python
# 맵 파일 경로 확인
map_file_path = os.path.join(os.getcwd(), DefaultParams.MAP_FILE_PATH)

if not os.path.exists(map_file_path):
    print("Map file not found")
    sys.exit(0)

# 맵 매니저 초기화
map_manager = MapManager(map_file_path)
map_manager.initialize_all_layers()
```

**작업 내용**:
- 맵 파일 존재 여부 확인
- MapManager 인스턴스 생성
- 모든 맵 레이어 로딩 및 인덱싱

**소요 시간**: 약 1-5초 (맵 크기에 따라 다름)

### 2단계: 로그 파일 로딩

```python
print(f"\n1. Loading log file: {log_file_path}")
log_loader = JsonLogLoader(
    log_file_path, 
    [DefaultParams.MAP_ORIGIN_X, DefaultParams.MAP_ORIGIN_Y]
)

parsed_entries = log_loader.get_parsed_entries(map_manager)
ego_states = log_loader.get_ego_states(parsed_entries)
dynamic_agents = log_loader.get_dynamic_agents(parsed_entries)

print(f"   Total entries: {len(parsed_entries)}")
```

**작업 내용**:
- JSON 파일 로딩
- 각 엔트리 파싱 (EgoState, TrackedObject 변환)
- Ego 상태 및 동적 에이전트 리스트 추출

**소요 시간**: 약 1-10초 (로그 크기에 따라 다름)

**출력 예시**:
```
1. Loading log file: dataset/observation_log_20250307-094118.json
   Total entries: 50000
```

### 3단계: 윈도우 추출

```python
print(f"\n2. Extracting 101-epoch windows")
print(f"   History: {history_epochs}, Current: 1, Future: {future_epochs}")

history_epochs = 40
future_epochs = 60

windows = []
start_idx = history_epochs
end_idx = len(parsed_entries) - future_epochs

if start_idx >= end_idx:
    print(f"ERROR: Insufficient data. Need at least {history_epochs + 1 + future_epochs} entries.")
    return

# 유효한 시나리오 수 계산
total_valid_range = end_idx - start_idx
valid_count = (total_valid_range + step_size - 1) // step_size

if max_scenarios > 0:
    valid_count = min(valid_count, max_scenarios)

print(f"   Valid scenarios (step_size={step_size}): {valid_count}")

for i in range(valid_count):
    center_idx = start_idx + i * step_size
    
    if center_idx >= end_idx:
        break
    
    window_start = center_idx - history_epochs
    window_end = center_idx + future_epochs + 1
    
    window = ScenarioWindow(
        center_idx=center_idx,
        center_timestamp=parsed_entries[center_idx].timestamp_us,
        ego_history=ego_states[window_start:center_idx],
        ego_current=ego_states[center_idx],
        ego_future=ego_states[center_idx+1:window_end],
        agents_history=dynamic_agents[window_start:center_idx],
        agents_current=dynamic_agents[center_idx],
        agents_future=dynamic_agents[center_idx+1:window_end],
    )
    
    windows.append(window)
```

**작업 내용**:
- 유효한 윈도우 범위 계산
- `step_size`에 따른 샘플링
- 각 윈도우에 대해 ScenarioWindow 객체 생성

**윈도우 범위**:
- 시작 인덱스: `history_epochs` (40) 이상
- 종료 인덱스: `len(entries) - future_epochs` (60) 미만
- 중심 인덱스: `start_idx <= center_idx < end_idx`

**샘플링 예시**:
- `step_size=1`: 모든 가능한 윈도우 생성 (예: 100, 101, 102, ...)
- `step_size=100`: 100 프레임마다 윈도우 생성 (예: 100, 200, 300, ...)

**소요 시간**: 약 1-5초 (윈도우 수에 따라 다름)

**출력 예시**:
```
2. Extracting 101-epoch windows
   History: 40, Current: 1, Future: 60
   Valid scenarios (step_size=100): 498
   Scenario indices: 40, 140, 240, ...
```

### 4단계: 시나리오 분류

```python
print(f"\n3. Classifying scenarios")
labeler = ScenarioLabeler(map_manager=map_manager, hz=20.0)

for i, window in enumerate(windows):
    if (i + 1) % 100 == 0:
        print(f"   Processed {i+1}/{len(windows)}...")
    labeler.classify(window)

print(f"   Classification complete!")
```

**작업 내용**:
- ScenarioLabeler 인스턴스 생성
- 각 윈도우에 대해 4단계 분류 파이프라인 실행
- 라벨 할당

**분류 단계**:
1. State-based Classification
2. Behavior-based Classification
3. Interaction-based Classification
4. Dynamics-based Classification

**소요 시간**: 약 0.1-1초/시나리오 (맵 쿼리 포함)

**출력 예시**:
```
3. Classifying scenarios
   Processed 100/498...
   Processed 200/498...
   ...
   Classification complete!
```

### 5단계: JSON 출력

```python
print(f"\n4. Exporting to JSON")
ScenarioExporter.export_batch(windows, output_dir_json)
```

**작업 내용**:
- 각 시나리오를 JSON 파일로 저장
- 요약 파일 (scenarios_summary.json) 생성

**출력 파일**:
- `scenario_000040.json`, `scenario_000140.json`, ...
- `scenarios_summary.json`

**소요 시간**: 약 0.01-0.05초/시나리오

**출력 예시**:
```
4. Exporting to JSON

Exported 498 scenarios to: ./labeled_scenarios
```

### 6단계: 시각화 (선택적)

```python
if visualize:
    print(f"\n5. Creating visualizations")
    os.makedirs(output_dir_viz, exist_ok=True)
    
    print("   Initializing CustomScenarioVisualizer...")
    visualizer = CustomScenarioVisualizer(map_manager=map_manager)
    
    success_count = 0
    for i, window in enumerate(windows):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(windows)} visualizations... (Success: {success_count})")
        
        viz_path = os.path.join(output_dir_viz, f"scenario_{window.center_idx:06d}.png")
        img = visualizer.visualize(window, save_path=viz_path)
        
        if img is not None:
            success_count += 1
    
    print(f"   Visualization complete: {success_count}/{len(windows)} successful")
```

**작업 내용**:
- CustomScenarioVisualizer 인스턴스 생성
- 각 시나리오 시각화
- PNG 이미지 저장

**소요 시간**: 약 0.1-0.5초/시나리오

**출력 예시**:
```
5. Creating visualizations
   Initializing CustomScenarioVisualizer...
   Processed 100/498 visualizations... (Success: 100)
   Processed 200/498 visualizations... (Success: 200)
   ...
   Visualization complete: 498/498 successful
```

### 7단계: 통계 생성

```python
print(f"\n6. Statistics")
print("="*80)

label_counts = {}
for window in windows:
    for label in window.labels:
        label_counts[label.label] = label_counts.get(label.label, 0) + 1

print(f"\nTop 20 Most Frequent Labels:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    percentage = 100 * count / len(windows)
    print(f"   {label:50s}: {count:4d} ({percentage:5.1f}%)")

avg_labels = np.mean([len(window.labels) for window in windows])
print(f"\nAverage labels per scenario: {avg_labels:.2f}")

print("\n" + "="*80)
print(f"Results saved:")
print(f"  - JSON: {output_dir_json}")
if visualize:
    print(f"  - Images: {output_dir_viz}")
print("="*80)
```

**작업 내용**:
- 라벨 분포 통계 계산
- 상위 20개 라벨 출력
- 평균 라벨 수 계산

**출력 예시**:
```
6. Statistics
================================================================================

Top 20 Most Frequent Labels:
   low_magnitude_speed                              :  450 ( 90.4%)
   following_lane_without_lead                      :  420 ( 84.3%)
   medium_magnitude_speed                           :  380 ( 76.3%)
   ...

Average labels per scenario: 3.45

================================================================================
Results saved:
  - JSON: ./labeled_scenarios
  - Images: ./imgs
================================================================================
```

## 실행 예시

### 기본 실행

```python
from Start import process_log_file
import DefaultParams

process_log_file(
    log_file_path=DefaultParams.LOG_FILE_NAME,
    map_manager=None,
    output_dir_json=DefaultParams.OUTPUT_DIR_SCENARIO_FILE,
    output_dir_viz=DefaultParams.OUTPUT_DIR_IMAGE_FILE,
    visualize=DefaultParams.VISUALIZE,
    max_scenarios=DefaultParams.MAX_SCENARIOS,
    step_size=DefaultParams.STEP_SIZE
)
```

### 커스텀 파라미터

```python
process_log_file(
    log_file_path="dataset/my_log.json",
    output_dir_json="./my_scenarios",
    output_dir_viz="./my_images",
    visualize=True,
    max_scenarios=100,  # 최대 100개만 처리
    step_size=50        # 50 프레임마다 샘플링
)
```

## 성능 최적화

### 샘플링

`step_size`를 조정하여 처리할 시나리오 수를 제어할 수 있습니다:

- `step_size=1`: 모든 시나리오 (가장 정확하지만 느림)
- `step_size=100`: 100 프레임마다 샘플링 (빠르지만 덜 정확)
- `step_size=1000`: 1000 프레임마다 샘플링 (매우 빠름)

### 최대 시나리오 수 제한

`max_scenarios`로 처리할 최대 시나리오 수를 제한할 수 있습니다:

- `max_scenarios=-1`: 전체 처리
- `max_scenarios=100`: 최대 100개만 처리

### 시각화 비활성화

시각화는 시간이 많이 걸리므로, JSON만 필요한 경우 비활성화할 수 있습니다:

```python
process_log_file(
    ...,
    visualize=False  # 시각화 비활성화
)
```

## 에러 처리

### 데이터 부족

윈도우 생성에 필요한 데이터가 부족한 경우:

```python
if start_idx >= end_idx:
    print(f"ERROR: Insufficient data. Need at least {history_epochs + 1 + future_epochs} entries.")
    return
```

### 맵 파일 없음

맵 파일이 없는 경우:

```python
if not os.path.exists(map_file_path):
    print("Map file not found")
    sys.exit(0)
```

### 시각화 실패

시각화 중 에러가 발생해도 다른 시나리오 처리를 계속합니다:

```python
img = visualizer.visualize(window, save_path=viz_path)
if img is not None:
    success_count += 1
```

## 전체 실행 시간

일반적인 실행 시간 (50,000 프레임 로그 기준):

- **맵 초기화**: 1-5초
- **로그 로딩**: 1-10초
- **윈도우 추출**: 1-5초
- **분류** (500개 시나리오): 50-500초
- **JSON 출력**: 5-25초
- **시각화** (500개 시나리오): 50-250초
- **총 시간**: 약 2-15분

## 메모리 사용량

- **맵 데이터**: 약 100-500MB
- **로그 데이터**: 약 50-200MB (50,000 프레임 기준)
- **윈도우 데이터**: 약 100-500MB (500개 윈도우 기준)
- **총 메모리**: 약 250MB-1.2GB

## 다음 단계

- [시스템 아키텍처 문서](./01_architecture.md) - 전체 시스템 구조
- [시나리오 분류기 문서](./03_scenario_labeler.md) - 분류 알고리즘 상세
- [데이터 출력 문서](./07_export.md) - JSON 출력 형식 상세

