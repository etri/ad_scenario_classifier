"""JSON 로그 파일을 NuPlan 데이터 구조로 변환하는 로더 모듈."""
from __future__ import annotations
# pylint: disable=unsubscriptable-object

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union

from ObjectTypes import LogEntry
from TrafficLightDataTypes import TrafficLightStatusData, TrafficLightStatusType

# 유틸리티 모듈
from CoordinateUtils import local_to_utm
from MathUtils import enu_to_euclidean_heading
from FileUtils import get_absolute_path

# NuPlan 프레임워크의 차량 상태 및 객체 관련 타입들
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.utils.helpers import get_unique_incremental_track_id


class JsonLogLoader:
    """JSON 로그 파일을 읽어 NuPlan 데이터 구조로 변환하는 로더."""
    
    def __init__(self, log_file_path: str, mapOrigin):
        """
        Args:
            log_file_path: JSON 로그 파일 경로
            mapOrigin: 맵 원점 좌표 [x, y] (UTM)
        """
        self.log_file_path = log_file_path
        self.data = None
        self.MAP_ORIGIN_X = float(mapOrigin[0])
        self.MAP_ORIGIN_Y = float(mapOrigin[1])
        
        self.category_to_tracked_object_type = {
            0: TrackedObjectType.VEHICLE,
            1: TrackedObjectType.PEDESTRIAN,
            2: TrackedObjectType.BICYCLE,
            3: TrackedObjectType.TRAFFIC_CONE,
            4: TrackedObjectType.BARRIER,
            5: TrackedObjectType.CZONE_SIGN,
            6: TrackedObjectType.GENERIC_OBJECT,
        }
    
    def load(self) -> List[Dict[str, Any]]:
        """JSON 로그 파일을 로드하여 파싱. 실패 시 빈 리스트 반환."""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            return self.data
        except FileNotFoundError:
            abs_path = get_absolute_path(self.log_file_path)
            print(f"Error: Log file not found at {self.log_file_path}")
            print(f"   Absolute path: {abs_path}")
            return []
        except PermissionError as e:
            abs_path = get_absolute_path(self.log_file_path)
            print(f"Error: Permission denied when reading log file: {self.log_file_path}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Details: {e}")
            return []
        except OSError as e:
            abs_path = get_absolute_path(self.log_file_path)
            print(f"Error: OS error when reading log file: {self.log_file_path}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Details: {e}")
            return []
        except UnicodeDecodeError as e:
            abs_path = get_absolute_path(self.log_file_path)
            print(f"Error: Encoding error when reading log file: {self.log_file_path}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Expected encoding: utf-8")
            print(f"   Details: {e}")
            return []
        except json.JSONDecodeError as e:
            abs_path = get_absolute_path(self.log_file_path)
            print(f"Error: Invalid JSON format in {self.log_file_path}")
            print(f"   Absolute path: {abs_path}")
            print(f"   Line {e.lineno}, Column {e.colno}: {e.msg}")
            return []
    
    def parse_ego_state(self, ego_data: Dict[str, Any], timestamp_us: int) -> EgoState:
        """
        Args:
            ego_data: Ego 차량 상태 데이터 (x, y, heading, v_x, v_y, acc_x, acc_y, steer_angle)
            timestamp_us: 타임스탬프 (마이크로초)
        
        Returns:
            EgoState 객체 (로컬 좌표 → UTM 변환, ENU → 유클리드 헤딩 변환 포함)
        """
        
        # 좌표 변환: 로컬 → UTM
        utm_x, utm_y = local_to_utm(
            ego_data['x'], 
            ego_data['y'], 
            self.MAP_ORIGIN_X, 
            self.MAP_ORIGIN_Y
        )
        
        # 헤딩 변환: ENU → 유클리드 좌표계
        heading_rad = enu_to_euclidean_heading(float(ego_data['heading']))
        
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(
                utm_x,    # UTM X 좌표
                utm_y,    # UTM Y 좌표
                heading_rad  # 유클리드 좌표계 헤딩 (라디안)
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
        
        
    def parse_traffic_light(self, ego_data: Dict[str, Any], map_api):
        """
        Args:
            ego_data: Ego 차량 데이터 (x, y, vehicle_traffic_light)
            map_api: MapManager 인스턴스
        
        Returns:
            (TrafficLightStatusType, int): (신호등 상태, 차선 연결부 ID)
        """
        # 로컬 좌표를 UTM 좌표로 변환
        x, y = local_to_utm(
            ego_data.get('x', 0.0),
            ego_data.get('y', 0.0),
            self.MAP_ORIGIN_X,
            self.MAP_ORIGIN_Y
        )
        
        # 신호등 상태 파싱
        tl_status = ego_data.get('vehicle_traffic_light', TrafficLightStatusType.UNKNOWN)
        if tl_status == 255:
            tl_status = TrafficLightStatusType.UNKNOWN
        
        ego_lane_id = int(map_api.get_nearest_lane((x, y))[0])
        return tl_status, ego_lane_id

    def parse_dynamic_agent(self, agent_data: Dict[str, Any], timestamp_us: Optional[int] = None) -> TrackedObject:
        """
        Args:
            agent_data: 에이전트 데이터 (agent_id, category, x, y, heading, width, length, height, v_x, v_y)
            timestamp_us: 타임스탬프 (마이크로초)
        
        Returns:
            TrackedObject (Agent 또는 StaticObject)
        """
        # 위치 및 헤딩으로부터 pose 생성 (좌표계 변환 포함)
        # 좌표 변환: 로컬 → UTM
        utm_x, utm_y = local_to_utm(
            agent_data.get('x', 0.0),
            agent_data.get('y', 0.0),
            self.MAP_ORIGIN_X,
            self.MAP_ORIGIN_Y
        )
        
        pose = StateSE2(
            utm_x,  # UTM X 좌표
            utm_y,  # UTM Y 좌표
            np.deg2rad(agent_data.get('heading', 0.0))  # 헤딩 각도 (라디안으로 변환)
        )
        
        oriented_box = OrientedBox(
            pose, 
            width=agent_data.get('width', 0.0),
            length=agent_data.get('length', 0.0),
            height=agent_data.get('height', 0.0)
        )
        
        category = agent_data.get('category', 0)
        tracked_object_type = self.category_to_tracked_object_type.get(category, TrackedObjectType.VEHICLE)
        
        agent_id = agent_data.get('agent_id', 0)
        track_token = f"agent_{agent_id:08d}"
        
        metadata = SceneObjectMetadata(
            token=f"token_{agent_id:08d}_{timestamp_us or 0}",
            track_token=track_token,
            track_id=get_unique_incremental_track_id(track_token),
            timestamp_us=timestamp_us or 0,
            category_name=tracked_object_type.name.lower(),
        )
        
        if tracked_object_type in AGENT_TYPES:
            return Agent(
                tracked_object_type=tracked_object_type,
                oriented_box=oriented_box,
                velocity=StateVector2D(agent_data.get('v_x', 0.0), agent_data.get('v_y', 0.0)),
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
    
    def parse_log_entry(self, entry_data: Dict[str, Any], map_api) -> LogEntry:
        """
        Args:
            entry_data: JSON 로그 엔트리 (timestamp_us, ego_state, dynamic_agents)
            map_api: MapManager 인스턴스
        
        Returns:
            LogEntry 객체
        """
        timestamp_us = entry_data.get('timestamp_us', 0)
        ego_state = self.parse_ego_state(entry_data.get('ego_state', {}), timestamp_us)
        traffic_light_status, ego_lane_id = self.parse_traffic_light(entry_data.get('ego_state', {}), map_api)
        
        dynamic_agents = []
        for agent_data in entry_data.get('dynamic_agents', []):
            dynamic_agents.append(self.parse_dynamic_agent(agent_data, timestamp_us))

        return LogEntry(
            timestamp_us=timestamp_us,
            ego_state=ego_state,
            dynamic_agents=dynamic_agents,
            traffic_light_status=TrafficLightStatusData(traffic_light_status, ego_lane_id, timestamp_us)
        )
    
    def get_parsed_entries(self, map_api = None) -> List[LogEntry]:
        """
        Args:
            map_api: MapManager 인스턴스 (선택적)
        
        Returns:
            파싱된 LogEntry 리스트
        """
        if self.data is None:
            self.load()
        
        parsed_entries = []
        for entry_data in (self.data or []):
            parsed_entries.append(self.parse_log_entry(entry_data, map_api))
        
        return parsed_entries
    
    def get_ego_states(self, parsed_entries) -> List[EgoState]:
        """LogEntry 리스트에서 EgoState 리스트 추출."""
        return [entry.ego_state for entry in parsed_entries]
    
    def get_dynamic_agents(self, parsed_entries) -> List[List[TrackedObject]]:
        """LogEntry 리스트에서 주변 객체 리스트 추출."""
        return [entry.dynamic_agents for entry in parsed_entries]
    
