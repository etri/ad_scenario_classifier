"""시나리오 분류 파이프라인 메인 실행 파일."""
from __future__ import annotations
# pylint: disable=unsubscriptable-object

import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, asdict
from DataTypes import ScenarioLabel, ScenarioWindow, LabeledScenario
import DefaultParams

from MapManager import MapManager
from JsonLogLoader import JsonLogLoader

# 유틸리티 모듈
from CoordinateUtils import world_to_ego_centric
from MathUtils import calculate_speed, calculate_distance, normalize_angle
from FileUtils import ensure_directory

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType



class ScenarioLabeler:
    """Rule-based 4단계 시나리오 분류 파이프라인."""
    
    SPEED_LOW_THRESHOLD = 2.78
    SPEED_MEDIUM_THRESHOLD = 11.11
    STATIONARY_SPEED_THRESHOLD = 0.1
    STATIONARY_DURATION = 0.5
    
    STOPLINE_DISTANCE_THRESHOLD = 2.0
    INTERSECTION_DISTANCE_THRESHOLD = 5.0
    
    HIGH_SPEED_VEHICLE_THRESHOLD = 20.0
    CONSTRUCTION_ZONE_DISTANCE = 20.0
    TRAFFIC_CONE_DISTANCE = 10.0
    
    TURN_HEADING_THRESHOLD = 15.0
    HIGH_SPEED_TURN_THRESHOLD = 8.0
    
    LEAD_VEHICLE_DISTANCE = 20.0
    SLOW_LEAD_SPEED_DIFF = 2.0
    
    JERK_THRESHOLD = 10.0
    LATERAL_ACCELERATION_THRESHOLD = 2.5
    
    MULTIPLE_VEHICLES_THRESHOLD = 10
    MULTIPLE_PEDESTRIANS_THRESHOLD = 3
    
    LONG_VEHICLE_LENGTH = 8.0
    
    def __init__(self, map_manager=None, hz: float = 20.0):
        """
        Args:
            map_manager: 맵 데이터 관리자 인스턴스
            hz: 데이터 샘플링 주파수
        """
        self.map_manager = map_manager
        self.hz = hz
        self.dt = 1.0 / hz
        
        self.confidence_levels = {
            "low_magnitude_speed": 0.99,
            "medium_magnitude_speed": 0.99,
            "high_magnitude_speed": 0.99,
            "stationary": 0.98,
            "stationary_in_traffic": 0.95,
            "on_stopline_traffic_light": 0.95,
            "on_stopline_stop_sign": 0.95,
            "on_stopline_crosswalk": 0.90,
            "on_intersection": 0.95,
            "on_traffic_light_intersection": 0.95,
            "on_all_way_stop_intersection": 0.90,
            "near_high_speed_vehicle": 0.85,
            "near_construction_zone_sign": 0.90,
            "near_trafficcone_on_driveable": 0.85,
            "near_barrier_on_driveable": 0.85,
            "near_long_vehicle": 0.90,
            "near_multiple_vehicles": 0.95,
            "near_multiple_pedestrians": 0.95,
            "starting_left_turn": 0.85,
            "starting_right_turn": 0.85,
            "starting_high_speed_turn": 0.80,
            "starting_low_speed_turn": 0.80,
            "changing_lane": 0.80,
            "changing_lane_to_left": 0.80,
            "changing_lane_to_right": 0.80,
            "following_lane_with_lead": 0.85,
            "following_lane_with_slow_lead": 0.80,
            "following_lane_without_lead": 0.90,
            "behind_long_vehicle": 0.85,
            "behind_bike": 0.85,
            "high_magnitude_jerk": 0.95,
            "high_lateral_acceleration": 0.95,
        }
        
        self.label_categories = {
            "speed_profile": ["low_magnitude_speed", "medium_magnitude_speed", "high_magnitude_speed"],
            "stationary": ["stationary", "stationary_in_traffic"],
            "turning": ["starting_left_turn", "starting_right_turn", "starting_high_speed_turn", "starting_low_speed_turn"],
            "lane_change": ["changing_lane", "changing_lane_to_left", "changing_lane_to_right"],
            "following": ["following_lane_with_lead", "following_lane_with_slow_lead", "following_lane_without_lead"],
            "proximity": ["near_high_speed_vehicle", "near_long_vehicle", "near_multiple_vehicles",
                         "near_construction_zone_sign", "near_trafficcone_on_driveable", "near_barrier_on_driveable", 
                         "behind_long_vehicle", "behind_bike", "near_multiple_pedestrians"],
            "dynamics": ["high_magnitude_jerk", "high_lateral_acceleration"],
        }
    
    def classify(self, window: ScenarioWindow) -> ScenarioWindow:
        """
        Args:
            window: 101-epoch 시나리오 윈도우
        
        Returns:
            라벨링된 시나리오 윈도우
        """
        labels = set()
        
        labels.update(self._classify_explicit_states(window))
        labels.update(self._classify_behaviors(window))
        labels.update(self._classify_interactions(window))
        labels.update(self._classify_dynamics(window))
        
        window.labels = [
            ScenarioLabel(
                label=label,
                confidence=self.confidence_levels.get(label, 0.5),
                category=self._get_label_category(label)
            )
            for label in sorted(labels)
        ]
        
        return window
    
    def _classify_explicit_states(self, window: ScenarioWindow) -> Set[str]:
        """명시적 상태 기반 분류 (속도, 정지, 근접 객체)."""
        labels = set()
        
        ego = window.ego_current
        agents = window.agents_current
        
        speed = self._get_speed(ego)
        if speed < self.SPEED_LOW_THRESHOLD:
            labels.add("low_magnitude_speed")
        elif speed < self.SPEED_MEDIUM_THRESHOLD:
            labels.add("medium_magnitude_speed")
        else:
            labels.add("high_magnitude_speed")
        
        if self._is_stationary(window.ego_history[-10:] + [ego]):
            labels.add("stationary")
            nearby_vehicles = self._count_nearby_vehicles(ego, agents, radius=30.0)
            if nearby_vehicles > 5:
                labels.add("stationary_in_traffic")
        
        labels.update(self._classify_proximity(ego, agents))
        
        return labels
    
    def _classify_behaviors(self, window: ScenarioWindow) -> Set[str]:
        """궤적 기반 행동 분류 (회전, 차선변경)."""
        labels = set()
        
        ego_history = window.ego_history
        ego_current = window.ego_current
        ego_future = window.ego_future
        
        turn_labels = self._detect_turning(ego_history[-20:], ego_current, ego_future[:20])
        labels.update(turn_labels)
        
        lane_change_labels = self._detect_lane_change(ego_history, ego_current, ego_future)
        labels.update(lane_change_labels)
        
        return labels
    
    def _classify_interactions(self, window: ScenarioWindow) -> Set[str]:
        """상호작용 기반 분류 (선행차 추종, 다중 객체)."""
        labels = set()
        
        ego = window.ego_current
        agents = window.agents_current
        
        lead_vehicle = self._get_lead_vehicle(ego, agents)
        
        if lead_vehicle:
            labels.add("following_lane_with_lead")
            
            ego_speed = self._get_speed(ego)
            lead_speed = self._get_agent_speed(lead_vehicle)
            
            if lead_speed < ego_speed - self.SLOW_LEAD_SPEED_DIFF:
                labels.add("following_lane_with_slow_lead")
            
            if self._is_long_vehicle(lead_vehicle):
                labels.add("behind_long_vehicle")
            elif lead_vehicle.tracked_object_type == TrackedObjectType.BICYCLE:
                labels.add("behind_bike")
        else:
            labels.add("following_lane_without_lead")
        
        vehicles = [a for a in agents if a.tracked_object_type == TrackedObjectType.VEHICLE]
        pedestrians = [a for a in agents if a.tracked_object_type == TrackedObjectType.PEDESTRIAN]
        
        if len(vehicles) > self.MULTIPLE_VEHICLES_THRESHOLD:
            labels.add("near_multiple_vehicles")
        
        if len(pedestrians) > self.MULTIPLE_PEDESTRIANS_THRESHOLD:
            labels.add("near_multiple_pedestrians")
        
        return labels
    
    def _classify_dynamics(self, window: ScenarioWindow) -> Set[str]:
        """동역학 분류 (jerk, 측면 가속도)."""
        labels = set()
        
        if len(window.ego_history) >= 2:
            recent_states = window.ego_history[-2:] + [window.ego_current]
            jerk = self._compute_jerk(recent_states)
            
            if abs(jerk) > self.JERK_THRESHOLD:
                labels.add("high_magnitude_jerk")
        
        if len(window.ego_history) >= 1:
            recent_states = window.ego_history[-1:] + [window.ego_current]
            lateral_acc = self._compute_lateral_acceleration(recent_states)
            
            if abs(lateral_acc) > self.LATERAL_ACCELERATION_THRESHOLD:
                labels.add("high_lateral_acceleration")
        
        return labels
    
    def _get_speed(self, ego: EgoState) -> float:
        """Ego 차량 속도 크기 계산."""
        vx = ego.dynamic_car_state.rear_axle_velocity_2d.x
        vy = ego.dynamic_car_state.rear_axle_velocity_2d.y
        return calculate_speed(vx, vy)
    
    def _get_agent_speed(self, agent: TrackedObject) -> float:
        """주변 객체 속도 크기 계산."""
        if hasattr(agent, 'velocity'):
            return calculate_speed(agent.velocity.x, agent.velocity.y)
        return 0.0
    
    def _is_stationary(self, ego_states: List[EgoState]) -> bool:
        """정지 상태 여부 판단."""
        if len(ego_states) < int(self.STATIONARY_DURATION * self.hz):
            return False
        speeds = [self._get_speed(state) for state in ego_states]
        return all(speed < self.STATIONARY_SPEED_THRESHOLD for speed in speeds)
    
    def _count_nearby_vehicles(self, ego: EgoState, agents: List[TrackedObject], radius: float) -> int:
        """
        주변 차량 수 계산.
        
        Args:
            ego (EgoState): Ego 차량 상태
            agents (List[TrackedObject]): 주변 객체 리스트
            radius (float): 검색 반경 (미터)
        
        Returns:
            int: 반경 내 차량 수
        """
        ego_pos = (ego.rear_axle.x, ego.rear_axle.y)
        count = 0
        for agent in agents:
            if agent.tracked_object_type == TrackedObjectType.VEHICLE:
                agent_pos = (agent.center.x, agent.center.y)
                if calculate_distance(ego_pos, agent_pos) < radius:
                    count += 1
        return count
    
    def _classify_proximity(self, ego: EgoState, agents: List[TrackedObject]) -> Set[str]:
        """근접 객체 분류."""
        labels = set()
        ego_pos = (ego.rear_axle.x, ego.rear_axle.y)
        
        for agent in agents:
            agent_pos = (agent.center.x, agent.center.y)
            distance = calculate_distance(ego_pos, agent_pos)
            
            if agent.tracked_object_type == TrackedObjectType.VEHICLE:
                speed = self._get_agent_speed(agent)
                if speed > self.HIGH_SPEED_VEHICLE_THRESHOLD and distance < 50.0:
                    labels.add("near_high_speed_vehicle")
                
                if self._is_long_vehicle(agent) and distance < 30.0:
                    labels.add("near_long_vehicle")
            
            elif agent.tracked_object_type == TrackedObjectType.CZONE_SIGN:
                if distance < self.CONSTRUCTION_ZONE_DISTANCE:
                    labels.add("near_construction_zone_sign")
            
            elif agent.tracked_object_type == TrackedObjectType.TRAFFIC_CONE:
                if distance < self.TRAFFIC_CONE_DISTANCE:
                    labels.add("near_trafficcone_on_driveable")
            
            elif agent.tracked_object_type == TrackedObjectType.BARRIER:
                if distance < self.TRAFFIC_CONE_DISTANCE:
                    labels.add("near_barrier_on_driveable")
        
        return labels
    
    def _is_long_vehicle(self, agent: TrackedObject) -> bool:
        """대형 차량 여부 판단."""
        return agent.box.length > self.LONG_VEHICLE_LENGTH
    
    def _detect_turning(self, ego_history: List[EgoState], ego_current: EgoState, 
                       ego_future: List[EgoState]) -> Set[str]:
        """헤딩 각도 변화 기반 회전 감지."""
        labels = set()
        
        if len(ego_history) < 5 or len(ego_future) < 5:
            return labels
        
        past_heading = ego_history[0].rear_axle.heading
        future_heading = ego_future[-1].rear_axle.heading if ego_future else ego_current.rear_axle.heading
        
        heading_change_rad = self._normalize_angle(future_heading - past_heading)
        heading_change_deg = math.degrees(heading_change_rad)
        
        if abs(heading_change_deg) > self.TURN_HEADING_THRESHOLD:
            if heading_change_deg > 0:
                labels.add("starting_left_turn")
            else:
                labels.add("starting_right_turn")
            
            speed = self._get_speed(ego_current)
            if speed > self.HIGH_SPEED_TURN_THRESHOLD:
                labels.add("starting_high_speed_turn")
            else:
                labels.add("starting_low_speed_turn")
        
        return labels
    
    def _normalize_angle(self, angle: float) -> float:
        return normalize_angle(angle)
    
    def _detect_lane_change(self, ego_history: List[EgoState], ego_current: EgoState,
                           ego_future: List[EgoState]) -> Set[str]:
        """횡방향 변위 기반 차선변경 감지."""
        labels = set()
        
        if len(ego_history) < 10 or len(ego_future) < 10:
            return labels
        
        past_positions = np.array([[s.rear_axle.x, s.rear_axle.y] for s in ego_history[-10:]])
        future_positions = np.array([[s.rear_axle.x, s.rear_axle.y] for s in ego_future[:10]])
        current_pos = np.array([ego_current.rear_axle.x, ego_current.rear_axle.y])
        
        future_vector = future_positions[-1] - current_pos
        past_heading = ego_history[-10].rear_axle.heading
        perpendicular = np.array([-math.sin(past_heading), math.cos(past_heading)])
        lateral_displacement = np.dot(future_vector, perpendicular)
        
        if abs(lateral_displacement) > 1.5:
            labels.add("changing_lane")
            
            if lateral_displacement > 0:
                labels.add("changing_lane_to_left")
            else:
                labels.add("changing_lane_to_right")
        
        return labels
    
    def _get_lead_vehicle(self, ego: EgoState, agents: List[TrackedObject]) -> Optional[TrackedObject]:
        """Ego 차량 앞 선행차 감지."""
        ego_pos = np.array([ego.rear_axle.x, ego.rear_axle.y])
        ego_heading = ego.rear_axle.heading
        ego_forward = np.array([math.cos(ego_heading), math.sin(ego_heading)])
        
        lead_vehicle = None
        min_distance = float('inf')
        
        for agent in agents:
            if agent.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            
            agent_pos = np.array([agent.center.x, agent.center.y])
            distance = calculate_distance(ego_pos, agent_pos)
            relative_pos = agent_pos - ego_pos
            
            forward_distance = np.dot(relative_pos, ego_forward)
            
            if forward_distance > 0 and distance < self.LEAD_VEHICLE_DISTANCE:
                lateral_distance = abs(np.cross(relative_pos, ego_forward))
                
                if lateral_distance < 2.0 and distance < min_distance:
                    min_distance = distance
                    lead_vehicle = agent
        
        return lead_vehicle
    
    def _compute_jerk(self, ego_states: List[EgoState]) -> float:
        """가속도 변화율(jerk) 계산."""
        if len(ego_states) < 3:
            return 0.0
        
        accels = []
        for state in ego_states:
            ax = state.dynamic_car_state.rear_axle_acceleration_2d.x
            ay = state.dynamic_car_state.rear_axle_acceleration_2d.y
            accel_mag = math.sqrt(ax**2 + ay**2)
            accels.append(accel_mag)
        
        jerk = (accels[-1] - accels[0]) / ((len(accels) - 1) * self.dt)
        return jerk
    
    def _compute_lateral_acceleration(self, ego_states: List[EgoState]) -> float:
        """측면 가속도 계산."""
        if len(ego_states) < 2:
            return 0.0
        
        state = ego_states[-1]
        speed = self._get_speed(state)
        
        if len(ego_states) >= 2:
            heading_rate = (ego_states[-1].rear_axle.heading - 
                          ego_states[-2].rear_axle.heading) / self.dt
            lateral_acc = speed * heading_rate
            return lateral_acc
        
        return 0.0
    
    def _get_label_category(self, label: str) -> str:
        """라벨의 카테고리 반환."""
        for category, labels in self.label_categories.items():
            if label in labels:
                return category
        return "other"


class CustomScenarioVisualizer:
    """Ego 중심 좌표계 기반 시나리오 시각화 컴포넌트."""
    
    def __init__(self, map_manager=None):
        """
        Args:
            map_manager: MapManager 인스턴스 (선택적)
        """
        self.map_manager = map_manager

        self.bounds = 45  # 2배 줌 적용 (기존 60에서 30으로 변경)
        self.offset = 15  # 비례 조정 (기존 20에서 10으로 변경)
        self.figsize = (15, 13)
        
        self.agent_colors = {
            TrackedObjectType.VEHICLE: "#001eff",
            TrackedObjectType.PEDESTRIAN: "#9500ff", 
            TrackedObjectType.BICYCLE: "#ff0059",
            TrackedObjectType.EGO: "#ff7f0e"
        }
        
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        self.ego_params = get_pacifica_parameters()
        
        self.origin = None
        self.angle = None
        self.rot_mat = None
        
        print("   CustomScenarioVisualizer initialized successfully")
    
    def visualize(self, window: ScenarioWindow, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Args:
            window: 시각화할 시나리오 윈도우
            save_path: 이미지 저장 경로 (선택적)
        
        Returns:
            시각화된 이미지 (RGB, uint8) 또는 None
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            self._setup_coordinate_system(window.ego_current)
            
            self._render_map(ax)
            self._render_trajectories(ax, window)
            self._render_agents(ax, window.agents_current)
            self._render_ego_vehicle(ax, window.ego_current)
            self._render_scenario_labels(ax, window.labels)
            
            self._configure_axes(ax)
            
            img = self._convert_to_image(fig)
            
            if img is not None and save_path:
                self._save_image(img, save_path)
            
            plt.close(fig)
            return img
            
        except Exception as e:  # pylint: disable=broad-except
            import traceback
            print(f"Warning: Custom visualization failed for scenario {window.center_idx}")
            print(f"   Error: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def _setup_coordinate_system(self, ego_state: EgoState):
        """Ego 중심 좌표계 설정."""
        self.origin = ego_state.rear_axle.array
        self.angle = ego_state.rear_axle.heading
        self.rot_mat = np.array([
            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ], dtype=np.float64)
    
    def _transform_coordinates(self, points):
        """
        월드 좌표를 Ego 중심 좌표로 변환.
        
        Args:
            points (np.ndarray): 변환할 점들
                형태: (N, 2) 또는 (2,) - (x, y) 좌표
        
        Returns:
            np.ndarray: 변환된 좌표 (Ego 중심 좌표계)
                Ego 차량이 원점 (0, 0)에 위치하고 전방이 +X 방향
        
        변환 공식:
            1. 평행이동: points - ego_origin
            2. 회전: rot_mat @ (points - ego_origin)
        """
        return world_to_ego_centric(points, self.origin, self.angle)
    
    def _render_map(self, ax):
        """맵 요소 렌더링 (차선, 횡단보도)."""
        if self.map_manager is None:
            return
            
        try:
            query_point = Point2D(self.origin[0], self.origin[1])
            road_elements = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            
            road_objects = self.map_manager.get_proximal_map_objects(
                query_point, self.bounds + self.offset, road_elements
            )
            
            all_lanes = (road_objects[SemanticMapLayer.LANE] + 
                        road_objects[SemanticMapLayer.LANE_CONNECTOR])
            
            for lane in all_lanes:
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
                    hatch="///"
                )
                ax.add_patch(crosswalk_patch)
                
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Map rendering failed: {e}")
    
    def _render_ego_vehicle(self, ax, ego_state: EgoState):
        """Ego 차량 렌더링."""
        try:
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
            
            length_indicator = self.ego_params.length * 0.75
            ax.plot(
                [1.69, 1.69 + length_indicator],
                [0, 0],
                color=self.agent_colors[TrackedObjectType.EGO],
                linewidth=2,
                zorder=11
            )
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Ego vehicle rendering failed: {e}")
    
    def _render_agents(self, ax, agents: List[TrackedObject]):
        """주변 객체 렌더링 (차량, 보행자, 자전거 등)."""
        for agent in agents:
            try:
                center = self._transform_coordinates(agent.center.array.reshape(1, -1))[0]
                angle = agent.center.heading - self.angle
                
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
                
                if hasattr(agent, 'velocity') and agent.velocity is not None:
                    velocity_magnitude = np.linalg.norm(agent.velocity.array)
                    if velocity_magnitude > 0.3:
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
                        
            except Exception as e:  # pylint: disable=broad-except
                print(f"   Warning: Agent rendering failed: {e}")
    
    def _render_trajectories(self, ax, window: ScenarioWindow):
        """Ego 및 주변 객체의 궤적 렌더링 (과거 및 미래)."""
        try:
            if window.ego_history:
                history_points = np.array([state.rear_axle.array for state in window.ego_history])
                transformed_history = self._transform_coordinates(history_points)
                
                ax.plot(
                    transformed_history[:, 0], 
                    transformed_history[:, 1],
                    color=self.agent_colors[TrackedObjectType.EGO],
                    alpha=0.5,
                    linewidth=5,
                    zorder=6,
                    label='Ego History'
                )
            
            if window.ego_future:
                future_points = np.array([state.rear_axle.array for state in window.ego_future])
                transformed_future = self._transform_coordinates(future_points)
                
                ax.plot(
                    transformed_future[:, 0], 
                    transformed_future[:, 1],
                    color="blue",
                    alpha=0.5,
                    linewidth=5,
                    zorder=6,
                    linestyle=":",
                    label='Ego Future'
                )
            
            agent_trajectories = self._extract_agent_trajectories(window)
            
            for agent_id, trajectory_data in agent_trajectories.items():
                if not trajectory_data:
                    continue
                    
                agent_type = trajectory_data.get('type', TrackedObjectType.VEHICLE)
                history_positions = trajectory_data.get('history', [])
                future_positions = trajectory_data.get('future', [])
                
                min_trajectory_length = 3
                
                current_pos = trajectory_data.get('current_position')
                if current_pos is None:
                    continue
                    
                base_color = self.agent_colors.get(agent_type, "gray")
                
                if len(history_positions) >= min_trajectory_length:
                    history_array = np.array(history_positions)
                    transformed_agent_history = self._transform_coordinates(history_array)
                    
                    ax.plot(
                        transformed_agent_history[:, 0],
                        transformed_agent_history[:, 1],
                        color=base_color,
                        alpha=0.3,
                        linewidth=5,
                        zorder=5,
                        linestyle='-'
                    )
                
                if len(future_positions) >= min_trajectory_length:
                    future_array = np.array(future_positions)
                    transformed_agent_future = self._transform_coordinates(future_array)
                    
                    ax.plot(
                        transformed_agent_future[:, 0],
                        transformed_agent_future[:, 1],
                        color=base_color,
                        alpha=0.25,
                        linewidth=5,
                        zorder=5,
                        linestyle='--'
                    )
                
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Trajectory rendering failed: {e}")
    
    def _extract_agent_trajectories(self, window: ScenarioWindow) -> Dict[str, Dict]:
        """시나리오 윈도우에서 주변 객체의 궤적 추출."""
        agent_trajectories = {}
        
        try:
            for agent in window.agents_current:
                agent_id = agent.track_token
                agent_trajectories[agent_id] = {
                    'type': agent.tracked_object_type,
                    'current_position': np.array([agent.center.x, agent.center.y]),
                    'history': [],
                    'future': []
                }
            
            for time_step, agents_at_time in enumerate(window.agents_history):
                for agent in agents_at_time:
                    agent_id = agent.track_token
                    if agent_id in agent_trajectories:
                        position = np.array([agent.center.x, agent.center.y])
                        agent_trajectories[agent_id]['history'].append(position)
            
            for time_step, agents_at_time in enumerate(window.agents_future):
                for agent in agents_at_time:
                    agent_id = agent.track_token
                    if agent_id in agent_trajectories:
                        position = np.array([agent.center.x, agent.center.y])
                        agent_trajectories[agent_id]['future'].append(position)
            
            return agent_trajectories
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Agent trajectory extraction failed: {e}")
            return {}
    
    def _render_scenario_labels(self, ax, labels: List[ScenarioLabel]):
        """시나리오 라벨 정보를 텍스트 오버레이로 렌더링."""
        if not labels:
            return
            
        try:
            categories = {}
            for label in labels:
                cat = label.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(label)
            
            text_lines = []
            for category, cat_labels in sorted(categories.items()):
                text_lines.append(f"{category.upper()}:")
                for label in cat_labels:
                    text_lines.append(f"  • {label.label} ({label.confidence:.2f})")
                text_lines.append("")
            
            full_text = "\n".join(text_lines).strip()
            
            if full_text:
                ax.text(
                    0.02, 0.98,
                    full_text,
                    transform=ax.transAxes,
                    fontsize=25,
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
                
        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Scenario labels rendering failed: {e}")
    
    def _configure_axes(self, ax):
        """Matplotlib 축 설정."""
        ax.set_xlim(xmin=-self.bounds + self.offset, xmax=self.bounds + self.offset)
        ax.set_ylim(ymin=-self.bounds + self.offset, ymax=self.bounds)
        ax.set_aspect('equal', adjustable='box')
        ax.axis("off")
        plt.tight_layout(pad=0)
    
    def _convert_to_image(self, fig) -> Optional[np.ndarray]:
        """Matplotlib figure를 numpy 배열로 변환."""
        try:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            
            # Handle different matplotlib versions
            try:
                buf = fig.canvas.buffer_rgba()
                img = np.asarray(buf).reshape(int(height), int(width), 4)
                img = img[:, :, :3]  # RGBA to RGB
            except AttributeError:
                try:
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                        int(height), int(width), 3
                    )
                except AttributeError:
                    fig.canvas.draw()
                    buf = fig.canvas.tostring_rgb()
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(
                        int(height), int(width), 3
                    )
            
            return img

        except Exception as e:  # pylint: disable=broad-except
            print(f"   Warning: Image conversion failed: {e}")
            return None
    
    def _save_image(self, img: np.ndarray, save_path: str):
        """이미지를 파일로 저장."""
        try:
            import cv2
            
            output_dir = os.path.dirname(save_path)
            if not ensure_directory(output_dir):
                return False
            
            try:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except cv2.error as e:
                print(f"   Warning: Failed to convert image color space: {e}")
                return False
            
            try:
                success = cv2.imwrite(save_path, img_bgr)
            except (cv2.error, OSError) as e:
                print(f"   Warning: cv2.imwrite raised exception for {save_path}: {e}")
                return False
            
            if not success:
                print(f"   Warning: cv2.imwrite failed for {save_path} (possibly disk full or permission issue)")
                return False
                
            if not os.path.exists(save_path):
                print(f"   Warning: File not created at {save_path}")
                return False
                
            return True

        except (OSError, IOError) as e:
            print(f"   Warning: Image saving failed: {e}")
            return False


class ScenarioExporter:
    """라벨링된 시나리오를 JSON 형식으로 출력하는 컴포넌트."""
    
    @staticmethod
    def export_scenario(window: ScenarioWindow, output_path: str):
        """
        Args:
            window: 출력할 시나리오 윈도우
            output_path: 출력 파일 경로
        """
        ego = window.ego_current
        speed = math.sqrt(
            ego.dynamic_car_state.rear_axle_velocity_2d.x**2 +
            ego.dynamic_car_state.rear_axle_velocity_2d.y**2
        )
        
        agents = window.agents_current
        num_vehicles = sum(1 for a in agents if a.tracked_object_type == TrackedObjectType.VEHICLE)
        num_pedestrians = sum(1 for a in agents if a.tracked_object_type == TrackedObjectType.PEDESTRIAN)
        
        labeled_scenario = LabeledScenario(
            scenario_id=f"scenario_{window.center_idx:06d}",
            center_idx=window.center_idx,
            center_timestamp=window.center_timestamp,
            ego_position={
                'x': float(ego.rear_axle.x),
                'y': float(ego.rear_axle.y),
                'heading': float(ego.rear_axle.heading)
            },
            ego_velocity={
                'vx': float(ego.dynamic_car_state.rear_axle_velocity_2d.x),
                'vy': float(ego.dynamic_car_state.rear_axle_velocity_2d.y),
                'magnitude': float(speed)
            },
            labels=[label.label for label in window.labels],
            label_details=[
                {
                    'label': label.label,
                    'confidence': label.confidence,
                    'category': label.category
                }
                for label in window.labels
            ],
            num_agents=len(agents),
            num_vehicles=num_vehicles,
            num_pedestrians=num_pedestrians,
            confidence_mean=float(np.mean([label.confidence for label in window.labels])) if window.labels else 0.0,
            categories=list(set(label.category for label in window.labels))
        )
        
        scenario_dict = asdict(labeled_scenario)
        
        observation_data = {
            'ego_history': [
                {
                    'timestamp': ego_state.timestamp_us if hasattr(ego_state, 'timestamp_us') else window.center_timestamp,
                    'position': {
                        'x': float(ego_state.rear_axle.x),
                        'y': float(ego_state.rear_axle.y),
                        'heading': float(ego_state.rear_axle.heading)
                    },
                    'velocity': {
                        'vx': float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
                        'vy': float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
                    },
                    'acceleration': {
                        'ax': float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x),
                        'ay': float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y)
                    }
                }
                for ego_state in window.ego_history
            ],
            'ego_current': {
                'timestamp': window.ego_current.timestamp_us if hasattr(window.ego_current, 'timestamp_us') else window.center_timestamp,
                'position': {
                    'x': float(window.ego_current.rear_axle.x),
                    'y': float(window.ego_current.rear_axle.y),
                    'heading': float(window.ego_current.rear_axle.heading)
                },
                'velocity': {
                    'vx': float(window.ego_current.dynamic_car_state.rear_axle_velocity_2d.x),
                    'vy': float(window.ego_current.dynamic_car_state.rear_axle_velocity_2d.y)
                },
                'acceleration': {
                    'ax': float(window.ego_current.dynamic_car_state.rear_axle_acceleration_2d.x),
                    'ay': float(window.ego_current.dynamic_car_state.rear_axle_acceleration_2d.y)
                }
            },
            'ego_future': [
                {
                    'timestamp': ego_state.timestamp_us if hasattr(ego_state, 'timestamp_us') else window.center_timestamp,
                    'position': {
                        'x': float(ego_state.rear_axle.x),
                        'y': float(ego_state.rear_axle.y),
                        'heading': float(ego_state.rear_axle.heading)
                    },
                    'velocity': {
                        'vx': float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
                        'vy': float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
                    },
                    'acceleration': {
                        'ax': float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x),
                        'ay': float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y)
                    }
                }
                for ego_state in window.ego_future
            ],
            'agents_history': [
                [
                    {
                        'id': agent.track_token,
                        'type': agent.tracked_object_type.name,
                        'position': {
                            'x': float(agent.center.x),
                            'y': float(agent.center.y),
                            'heading': float(agent.center.heading)
                        },
                        'velocity': {
                            'vx': float(agent.velocity.x) if hasattr(agent, 'velocity') and agent.velocity else 0.0,
                            'vy': float(agent.velocity.y) if hasattr(agent, 'velocity') and agent.velocity else 0.0
                        },
                        'box': {
                            'length': float(agent.box.length),
                            'width': float(agent.box.width),
                            'height': float(agent.box.height)
                        }
                    }
                    for agent in agents_at_time
                ]
                for agents_at_time in window.agents_history
            ],
            'agents_current': [
                {
                    'id': agent.track_token,
                    'type': agent.tracked_object_type.name,
                    'position': {
                        'x': float(agent.center.x),
                        'y': float(agent.center.y),
                        'heading': float(agent.center.heading)
                    },
                    'velocity': {
                        'vx': float(agent.velocity.x) if hasattr(agent, 'velocity') and agent.velocity else 0.0,
                        'vy': float(agent.velocity.y) if hasattr(agent, 'velocity') and agent.velocity else 0.0
                    },
                    'box': {
                        'length': float(agent.box.length),
                        'width': float(agent.box.width),
                        'height': float(agent.box.height)
                    }
                }
                for agent in window.agents_current
            ],
            'agents_future': [
                [
                    {
                        'id': agent.track_token,
                        'type': agent.tracked_object_type.name,
                        'position': {
                            'x': float(agent.center.x),
                            'y': float(agent.center.y),
                            'heading': float(agent.center.heading)
                        },
                        'velocity': {
                            'vx': float(agent.velocity.x) if hasattr(agent, 'velocity') and agent.velocity else 0.0,
                            'vy': float(agent.velocity.y) if hasattr(agent, 'velocity') and agent.velocity else 0.0
                        },
                        'box': {
                            'length': float(agent.box.length),
                            'width': float(agent.box.width),
                            'height': float(agent.box.height)
                        }
                    }
                    for agent in agents_at_time
                ]
                for agents_at_time in window.agents_future
            ],
            'traffic_light_status': window.traffic_light_status
        }
        
        scenario_dict['observation_data'] = observation_data
        
        try:
            output_dir = os.path.dirname(output_path)
            ensure_directory(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
        except PermissionError as e:
            print(f"Error: Permission denied when writing to {output_path}")
            print(f"   Details: {e}")
            raise
        except OSError as e:
            print(f"Error: OS error when writing to {output_path}")
            print(f"   Details: {e}")
            raise
        except (TypeError, ValueError) as e:
            print(f"Error: JSON serialization failed for scenario {window.center_idx}")
            print(f"   Details: {e}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error when exporting scenario to {output_path}")
            print(f"   Details: {e}")
            raise
    
    @staticmethod
    def export_batch(windows: List[ScenarioWindow], output_dir: str):
        """
        Args:
            windows: 출력할 시나리오 윈도우 리스트
            output_dir: 출력 디렉토리 경로
        """
        # Create output directory with exception handling
        if not ensure_directory(output_dir):
            raise OSError(f"Failed to create output directory: {output_dir}")
        
        success_count = 0
        error_count = 0
        for window in windows:
            try:
                filename = f"scenario_{window.center_idx:06d}.json"
                output_path = os.path.join(output_dir, filename)
                ScenarioExporter.export_scenario(window, output_path)
                success_count += 1
            except (OSError, IOError, TypeError, ValueError) as e:
                error_count += 1
                print(f"Warning: Failed to export scenario {window.center_idx}: {e}")
        
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
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except PermissionError as e:
            print(f"Error: Permission denied when writing summary file: {summary_path}")
            print(f"   Details: {e}")
            raise
        except OSError as e:
            print(f"Error: OS error when writing summary file: {summary_path}")
            print(f"   Details: {e}")
            raise
        except (TypeError, ValueError) as e:
            print(f"Error: JSON serialization failed for summary file")
            print(f"   Details: {e}")
            raise
        
        print(f"\nExported {success_count}/{len(windows)} scenarios to: {output_dir}")
        if error_count > 0:
            print(f"   Warning: {error_count} scenario(s) failed to export")


def process_log_file(log_file_path: str, 
                    map_manager=None,
                    output_dir_json: str = "./labeled_scenarios",
                    output_dir_viz: str = "./imgs",
                    visualize: bool = True,
                    max_scenarios: int = -1,
                    step_size: int = 1):
    """
    Args:
        log_file_path: JSON 로그 파일 경로
        map_manager: 맵 매니저 인스턴스 (None이면 자동 생성)
        output_dir_json: JSON 출력 디렉토리
        output_dir_viz: 시각화 이미지 출력 디렉토리
        visualize: 시각화 이미지 생성 여부
        max_scenarios: 처리할 최대 시나리오 수 (-1 = 전체)
        step_size: 시나리오 윈도우 생성 스텝 크기
    """
    sys.path.insert(0, os.path.dirname(log_file_path))
    
    map_file_path = os.path.join(os.getcwd(), DefaultParams.MAP_FILE_PATH)
    
    print(f"지도 파일 경로: {map_file_path}")
    
    # 지도 파일 검증 및 예외처리
    from FileUtils import validate_file_path
    validate_file_path(map_file_path, "Map file")
    
    map_manager = MapManager(map_file_path)
    map_manager.initialize_all_layers()

    print(f"\n1. Loading log file: {log_file_path}")
    log_loader = JsonLogLoader(log_file_path, [DefaultParams.MAP_ORIGIN_X, DefaultParams.MAP_ORIGIN_Y])
    
    parsed_entries = log_loader.get_parsed_entries(map_manager)
    ego_states = log_loader.get_ego_states(parsed_entries)
    dynamic_agents = log_loader.get_dynamic_agents(parsed_entries)
    
    print(f"   Total entries: {len(parsed_entries)}")
    
    history_epochs = 40
    future_epochs = 60
    
    print(f"\n2. Extracting 101-epoch windows")
    print(f"   History: {history_epochs}, Current: 1, Future: {future_epochs}")
    
    windows = []
    start_idx = history_epochs
    end_idx = len(parsed_entries) - future_epochs
    
    if start_idx >= end_idx:
        print(f"ERROR: Insufficient data. Need at least {history_epochs + 1 + future_epochs} entries.")
        return
    
    total_valid_range = end_idx - start_idx
    valid_count = (total_valid_range + step_size - 1) // step_size
    
    if max_scenarios > 0:
        valid_count = min(valid_count, max_scenarios)
    
    print(f"   Valid scenarios (step_size={step_size}): {valid_count}")
    print(f"   Scenario indices: {start_idx}, {start_idx + step_size}, {start_idx + 2*step_size}, ...")
    
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
    
    print(f"\n3. Classifying scenarios")
    labeler = ScenarioLabeler(map_manager=map_manager, hz=20.0)
    
    for i, window in enumerate(windows):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(windows)}...")
        labeler.classify(window)
    
    print(f"   Classification complete!")
    
    print(f"\n4. Exporting to JSON")
    ScenarioExporter.export_batch(windows, output_dir_json)
    
    if visualize:
        print(f"\n5. Creating visualizations")
        if not ensure_directory(output_dir_viz):
            raise OSError(f"Failed to create visualization directory: {output_dir_viz}")
        
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
        if success_count == 0:
            print("   ERROR: No visualizations were generated! Check the error messages above.")
    
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



if __name__ == "__main__":
    
    print(f"설정:")
    print(f"  - 로그 파일: {DefaultParams.LOG_FILE_NAME}")
    print(f"  - JSON 출력: {DefaultParams.OUTPUT_DIR_SCENARIO_FILE}")
    print(f"  - 이미지 출력: {DefaultParams.OUTPUT_DIR_IMAGE_FILE}")
    print(f"  - 최대 시나리오 수: {DefaultParams.MAX_SCENARIOS if DefaultParams.MAX_SCENARIOS > 0 else '전체'}")
    print(f"  - 시각화: {'활성화' if DefaultParams.VISUALIZE else '비활성화'}")
    print(f"  - 스텝 크기: {DefaultParams.STEP_SIZE} ({'전체 시나리오' if DefaultParams.STEP_SIZE == 1 else f'{DefaultParams.STEP_SIZE}개씩 건너뛰기'})")
    print()
    
    process_log_file(
        log_file_path=DefaultParams.LOG_FILE_NAME,
        map_manager=None,
        output_dir_json=DefaultParams.OUTPUT_DIR_SCENARIO_FILE,
        output_dir_viz=DefaultParams.OUTPUT_DIR_IMAGE_FILE,
        visualize=DefaultParams.VISUALIZE,
        max_scenarios=DefaultParams.MAX_SCENARIOS,
        step_size=DefaultParams.STEP_SIZE
    )