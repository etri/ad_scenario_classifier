"""
객체 타입 정의 모듈

이 모듈은 시나리오 분류기 시스템에서 사용되는 기본 객체 타입을 정의합니다.
주요 데이터 클래스:
- LogEntry: 단일 타임스탬프의 전체 상황 정보

참조:
    - doc/02_data_structures.md: 데이터 구조 상세 설명
    - doc/01_architecture.md: 데이터 로딩 레이어 설명
"""

from dataclasses import dataclass
from typing import List

from TrafficLightDataTypes import TrafficLightStatusData
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject

@dataclass
class LogEntry:
    """
    단일 타임스탬프에서의 전체 상황 정보를 나타내는 데이터 구조.
    
    JsonLogLoader가 JSON 로그 파일의 각 엔트리를 파싱하여 생성합니다.
    LogEntry 리스트는 시나리오 윈도우 추출의 기본 데이터로 사용됩니다.
    
    데이터 흐름:
        JSON 로그 파일
          └─> JsonLogLoader.parse_log_entry()
              └─> LogEntry
                  └─> List[LogEntry] (전체 로그)
                      └─> ScenarioWindow 추출 (101-epoch 윈도우)
    
    Attributes:
        timestamp_us (int): 타임스탬프 (마이크로초)
            Unix epoch 기준 마이크로초 단위의 타임스탬프입니다.
            로그 엔트리들의 시간 순서를 결정하는 기준이 됩니다.
            예: 1234567890000000 (2023년 2월 13일 23:31:30 UTC)
        
        ego_state (EgoState): Ego 차량의 상태 정보
            NuPlan 프레임워크의 EgoState 객체로, 다음 정보를 포함합니다:
            - rear_axle: 후축 위치 및 헤딩 (x, y, heading) - UTM 좌표계
            - dynamic_car_state: 동적 상태
              * rear_axle_velocity_2d: 속도 벡터 (vx, vy) - m/s
              * rear_axle_acceleration_2d: 가속도 벡터 (ax, ay) - m/s²
            - car_footprint: 차량 외곽선 (Shapely Polygon)
            - timestamp_us: 타임스탬프 (마이크로초)
            
            좌표계:
                - 위치: UTM 좌표계 (미터 단위)
                - 헤딩: 라디안 (0 = 동쪽, 반시계 방향이 양수)
                - 속도/가속도: m/s, m/s²
        
        dynamic_agents (List[TrackedObject]): 주변 동적 객체 리스트
            현재 시점에서 관측된 모든 주변 객체를 포함합니다.
            객체 타입:
            - TrackedObjectType.VEHICLE: 차량
            - TrackedObjectType.PEDESTRIAN: 보행자
            - TrackedObjectType.BICYCLE: 자전거
            - TrackedObjectType.TRAFFIC_CONE: 콘 (정적 객체)
            - TrackedObjectType.BARRIER: 장벽 (정적 객체)
            - TrackedObjectType.CZONE_SIGN: 공사구역 표지판 (정적 객체)
            
            각 TrackedObject는 다음 정보를 포함:
            - center: 중심 위치 및 헤딩 (x, y, heading) - UTM 좌표계
            - box: 바운딩 박스 (length, width, height) - 미터
            - velocity: 속도 벡터 (vx, vy) - m/s (Agent인 경우)
            - track_token: 객체 추적 토큰 (고유 ID)
        
        traffic_light_status (TrafficLightStatusData): 신호등 상태 정보
            Ego 차량과 가장 가까운 신호등의 상태 정보를 포함합니다.
            TrafficLightStatusData 객체로, 다음 정보를 포함:
            - status: 신호등 상태 (GO_STRAIGHT, STOP, WARNING 등)
            - lane_connector_id: 신호등이 속한 차선 연결부 ID
            - timestamp: 타임스탬프 (마이크로초)
            
            참조:
                - TrafficLightDataTypes: 신호등 데이터 타입 정의
                - doc/02_data_structures.md: TrafficLightStatusData 상세 설명
    
    사용 예시:
        # JsonLogLoader를 사용하여 LogEntry 리스트 생성
        loader = JsonLogLoader(log_file_path, map_origin)
        entries = loader.get_parsed_entries(map_api)
        
        # 각 LogEntry는 단일 타임스탬프의 전체 상황을 나타냄
        for entry in entries:
            print(f"Time: {entry.timestamp_us}")
            print(f"Ego position: ({entry.ego_state.rear_axle.x}, {entry.ego_state.rear_axle.y})")
            print(f"Number of agents: {len(entry.dynamic_agents)}")
        
        # ScenarioWindow 추출에 사용
        # (과거 40 + 현재 1 + 미래 60 = 101 프레임)
        window = extract_scenario_window(entries, center_idx=100)
    
    참조:
        - doc/02_data_structures.md: LogEntry 상세 설명
        - doc/01_architecture.md: 데이터 로딩 및 윈도우 추출 과정
        - JsonLogLoader: JSON 로그 파싱 및 LogEntry 생성
    """
    timestamp_us: int                            # 타임스탬프 (microsec, Unix epoch 기준)
    ego_state: EgoState                          # ego 상태 정보 (위치, 속도, 가속도, 헤딩 등)
    dynamic_agents: List[TrackedObject]          # 주변 동적 객체들 (차량, 보행자, 자전거 등)
    traffic_light_status: TrafficLightStatusData # 신호등 상태 정보 (가장 가까운 신호등)
