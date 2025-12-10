"""
신호등 데이터 타입 정의 모듈

이 모듈은 신호등 상태 정보를 나타내는 데이터 타입을 정의합니다.
주요 클래스:
- TrafficLightStatusType: 신호등 상태 열거형
- TrafficLightStatusData: 신호등 상태 데이터 구조

참조:
    - doc/02_data_structures.md: TrafficLightStatusData 상세 설명
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List


class TrafficLightStatusType(IntEnum):
    """
    신호등 상태를 나타내는 열거형.
    
    NuPlan 프레임워크의 신호등 상태를 확장하여 정의한 열거형입니다.
    각 상태는 정수 값과 이름을 가지며, JSON 직렬화/역직렬화를 지원합니다.
    
    상태 값:
        GO_STRAIGHT (0): 직진 가능
            신호등이 초록불이며, 직진이 허용된 상태입니다.
        
        GO_STRAIGHT_AND_TURNLEFT (1): 직진 및 좌회전 가능
            신호등이 초록불이며, 직진과 좌회전이 모두 허용된 상태입니다.
            일반적으로 좌회전 신호등이 있는 경우입니다.
        
        STOP (2): 정지
            신호등이 빨간불이며, 정지해야 하는 상태입니다.
            차량은 정지선에서 대기해야 합니다.
        
        STOP_AND_TURNLEFT (3): 정지 및 좌회전 가능
            직진은 정지해야 하지만, 좌회전은 허용된 상태입니다.
            일부 교차로에서 좌회전 전용 신호등이 있는 경우입니다.
        
        STOP_AND_WARNING (4): 정지 및 경고
            정지 상태이지만 경고 신호가 함께 표시된 상태입니다.
            곧 신호가 바뀔 예정임을 나타냅니다.
        
        WARNING (5): 경고
            신호등이 노란불(경고) 상태입니다.
            차량은 정지선을 넘지 않을 수 있다면 정지해야 합니다.
        
        UNKNOWN (6): 알 수 없음
            신호등 상태를 확인할 수 없는 상태입니다.
            신호등이 없거나, 데이터가 누락된 경우입니다.
    
    사용 예시:
        # 상태 확인
        if status == TrafficLightStatusType.STOP:
            # 정지해야 함
            pass
        
        # 직렬화 (JSON 저장용)
        status_str = TrafficLightStatusType.GO_STRAIGHT.serialize()
        # 결과: "GO_STRAIGHT"
        
        # 역직렬화 (JSON 로드용)
        status = TrafficLightStatusType.deserialize("GO_STRAIGHT")
        # 결과: TrafficLightStatusType.GO_STRAIGHT
    
    참조:
        - doc/02_data_structures.md: 신호등 데이터 구조 설명
    """
    # 확장된 신호등 상태 정의
    UNKNOWN = 6                     # 알 수 없음 (신호등 없음 또는 데이터 누락)
    GO_STRAIGHT = 0                 # 직진 가능 (초록불)
    GO_STRAIGHT_AND_TURNLEFT = 1    # 직진 및 좌회전 가능 (초록불 + 좌회전 신호)
    STOP = 2                        # 정지 (빨간불)
    STOP_AND_TURNLEFT = 3           # 정지 및 좌회전 가능 (빨간불 + 좌회전 전용 신호)
    STOP_AND_WARNING = 4            # 정지 및 경고 (빨간불 + 경고 신호)
    WARNING = 5                     # 경고 (노란불)

    def serialize(self) -> str:
        """
        신호등 상태를 문자열로 직렬화.
        
        JSON 저장 시 사용됩니다.
        
        Returns:
            str: 상태 이름 (예: "GO_STRAIGHT", "STOP")
        """
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """
        문자열을 신호등 상태로 역직렬화.
        
        JSON 로드 시 사용됩니다.
        
        Args:
            key: 상태 이름 (예: "GO_STRAIGHT", "STOP")
        
        Returns:
            TrafficLightStatusType: 해당하는 신호등 상태
        
        Raises:
            KeyError: 유효하지 않은 상태 이름인 경우
        """
        return TrafficLightStatusType.__members__[key]
    

@dataclass
class TrafficLightStatusData:
    """
    신호등 상태 정보를 나타내는 데이터 구조.
    
    Ego 차량과 가장 가까운 신호등의 상태 정보를 저장합니다.
    JsonLogLoader가 JSON 로그에서 신호등 정보를 파싱하여 생성하며,
    LogEntry의 traffic_light_status 필드에 저장됩니다.
    
    Attributes:
        status (TrafficLightStatusType): 신호등 상태
            GO_STRAIGHT, STOP, WARNING 등의 상태 값입니다.
            TrafficLightStatusType 열거형을 사용합니다.
        
        lane_connector_id (int): 신호등이 속한 차선 연결부 ID
            맵 데이터베이스의 차선 연결부(LaneConnector) ID입니다.
            신호등이 위치한 차선을 식별하는 데 사용됩니다.
            MapManager를 통해 해당 차선의 상세 정보를 조회할 수 있습니다.
        
        timestamp (int): 타임스탬프 (마이크로초 단위)
            신호등 상태가 관측된 시점의 타임스탬프입니다.
            Unix epoch 기준 마이크로초 단위입니다.
            LogEntry의 timestamp_us와 동일한 값입니다.
    
    사용 예시:
        # JsonLogLoader에서 생성
        tl_status, lane_id = loader.parse_traffic_light(ego_data, map_api)
        traffic_light_data = TrafficLightStatusData(tl_status, lane_id, timestamp_us)
        
        # 상태 확인
        if traffic_light_data.status == TrafficLightStatusType.STOP:
            # 정지해야 함
            pass
        
        # JSON 직렬화
        json_data = traffic_light_data.serialize()
        # 결과: {
        #     'status': 'STOP',
        #     'lane_connector_id': 12345,
        #     'timestamp': 1234567890000000
        # }
        
        # JSON 역직렬화
        data = TrafficLightStatusData.deserialize(json_data)
    
    참조:
        - doc/02_data_structures.md: TrafficLightStatusData 상세 설명
        - JsonLogLoader.parse_traffic_light(): 신호등 정보 파싱
        - LogEntry: 신호등 상태 정보가 포함된 로그 엔트리
    """
    status: TrafficLightStatusType  # 신호등 상태 (GO_STRAIGHT, STOP, WARNING 등)
    lane_connector_id: int          # 신호등이 속한 차선 연결부 ID (맵 데이터베이스 기준)
    timestamp: int                  # 타임스탬프 (마이크로초 단위, Unix epoch 기준)

    def serialize(self) -> Dict[str, Any]:
        """
        신호등 상태 데이터를 딕셔너리로 직렬화.
        
        JSON 저장 시 사용됩니다.
        
        Returns:
            Dict[str, Any]: 직렬화된 데이터
                {
                    'status': str,              # 상태 이름 (예: "STOP")
                    'lane_connector_id': int,   # 차선 연결부 ID
                    'timestamp': int            # 타임스탬프 (microsec)
                }
        """
        return {
            'status': self.status.serialize(),      # 상태를 문자열로 직렬화
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """
        딕셔너리를 신호등 상태 데이터로 역직렬화.
        
        JSON 로드 시 사용됩니다.
        
        Args:
            data: 직렬화된 데이터 딕셔너리
                {
                    'status': str,              # 상태 이름
                    'lane_connector_id': int,   # 차선 연결부 ID
                    'timestamp': int            # 타임스탬프
                }
        
        Returns:
            TrafficLightStatusData: 역직렬화된 신호등 상태 데이터
        
        Raises:
            KeyError: 필수 필드가 없는 경우
            ValueError: 상태 이름이 유효하지 않은 경우
        """
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data['status']),  # 문자열을 상태 타입으로 변환
            lane_connector_id=data['lane_connector_id'],
            timestamp=data['timestamp'],
        )


@dataclass
class TrafficLightStatuses:
    """
    여러 신호등 상태 정보를 담는 컨테이너 클래스.
    
    현재는 사용되지 않지만, 향후 확장을 위해 정의되어 있습니다.
    여러 신호등의 상태를 한 번에 관리할 때 사용할 수 있습니다.
    
    Attributes:
        traffic_lights (List[TrafficLightStatusData]): 신호등 상태 데이터 리스트
            여러 신호등의 상태 정보를 담는 리스트입니다.
    
    참조:
        - doc/02_data_structures.md: 신호등 데이터 구조 설명
    """
    traffic_lights: List[TrafficLightStatusData]  # 신호등 상태 데이터 리스트 (향후 확장용)
