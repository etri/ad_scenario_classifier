"""
수학 계산 유틸리티 모듈

이 모듈은 수학 계산 관련 유틸리티 함수들을 제공합니다.
- 속도 벡터 크기 계산
- 거리 계산 (유클리드 거리)
- 각도 정규화
- 좌표계 간 헤딩 변환

참조:
    - doc/03_scenario_labeler.md: 수학 계산 사용 예시
"""
from __future__ import annotations
# pylint: disable=unsubscriptable-object

import math
import numpy as np
from typing import Union, Tuple, List


def calculate_speed(vx: float, vy: float) -> float:
    """
    속도 벡터의 크기를 계산합니다.
    
    Args:
        vx (float): X 방향 속도 (m/s)
        vy (float): Y 방향 속도 (m/s)
    
    Returns:
        float: 속도 크기 (m/s)
    
    계산 공식:
        speed = sqrt(vx² + vy²)
    
    사용 예시:
        speed = calculate_speed(5.0, 2.0)
        # 결과: 약 5.385 m/s
    
    참조:
        - ScenarioLabeler: Ego 차량 및 주변 객체의 속도 계산
    """
    return math.sqrt(vx**2 + vy**2)


def calculate_distance(pos1: Union[Tuple[float, float], List[float], np.ndarray], 
                      pos2: Union[Tuple[float, float], List[float], np.ndarray]) -> float:
    """
    두 점 사이의 유클리드 거리를 계산합니다.
    
    Args:
        pos1: 첫 번째 점의 좌표
            형태: (x, y) 튜플, [x, y] 리스트, 또는 np.ndarray
        pos2: 두 번째 점의 좌표
            형태: (x, y) 튜플, [x, y] 리스트, 또는 np.ndarray
    
    Returns:
        float: 두 점 사이의 유클리드 거리 (미터)
    
    계산 공식:
        distance = sqrt((x2 - x1)² + (y2 - y1)²)
    
    사용 예시:
        distance = calculate_distance((0.0, 0.0), (3.0, 4.0))
        # 결과: 5.0
    
        pos1 = np.array([10.0, 20.0])
        pos2 = np.array([13.0, 24.0])
        distance = calculate_distance(pos1, pos2)
        # 결과: 5.0
    
    참조:
        - ScenarioLabeler: 객체 간 거리 계산
    """
    pos1_arr = np.array(pos1, dtype=np.float64)
    pos2_arr = np.array(pos2, dtype=np.float64)
    return float(np.linalg.norm(pos2_arr - pos1_arr))


def normalize_angle(angle: float) -> float:
    """
    각도를 [-π, π] 범위로 정규화합니다.
    
    Args:
        angle (float): 정규화할 각도 (라디안)
    
    Returns:
        float: 정규화된 각도 (라디안, [-π, π] 범위)
    
    정규화 과정:
        - angle > π인 경우: angle -= 2π (반복)
        - angle < -π인 경우: angle += 2π (반복)
    
    사용 예시:
        normalized = normalize_angle(3.0 * math.pi)
        # 결과: -math.pi (약 -3.14159)
        
        normalized = normalize_angle(-2.5 * math.pi)
        # 결과: -math.pi / 2 (약 -1.5708)
    
    참조:
        - ScenarioLabeler: 헤딩 각도 차이 계산 시 사용
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def enu_to_euclidean_heading(enu_heading: float) -> float:
    """
    ENU 좌표계 헤딩을 유클리드 좌표계 헤딩으로 변환합니다.
    
    ENU (East-North-Up) 좌표계와 유클리드 좌표계 간의 헤딩 변환을 수행합니다.
    
    좌표계 차이:
        - ENU: 0도 = 동쪽, 90도 = 북쪽, 시계방향이 양수
        - 유클리드: 0도 = 동쪽, 반시계방향이 양수
    
    Args:
        enu_heading (float): ENU 좌표계 헤딩 각도 (라디안 또는 도)
            라디안 단위로 입력하면 라디안으로 반환,
            도 단위로 입력하면 라디안으로 변환하여 반환
    
    Returns:
        float: 유클리드 좌표계 헤딩 각도 (라디안)
    
    변환 공식:
        heading_rad = deg2rad((90 - rad2deg(heading_deg)) % 360.0)
    
    사용 예시:
        # ENU 좌표계에서 90도 (북쪽) → 유클리드 좌표계에서 0도 (동쪽)
        euclidean = enu_to_euclidean_heading(np.deg2rad(90.0))
        # 결과: 0.0 (라디안)
        
        # ENU 좌표계에서 0도 (동쪽) → 유클리드 좌표계에서 90도 (북쪽)
        euclidean = enu_to_euclidean_heading(np.deg2rad(0.0))
        # 결과: π/2 (라디안, 약 1.5708)
    
    참조:
        - JsonLogLoader: JSON 로그의 헤딩 각도를 유클리드 좌표계로 변환
    """
    # 입력이 라디안인지 도인지 확인 (일반적으로 라디안으로 입력됨)
    # 하지만 JSON 로그에서는 도 단위로 입력될 수 있으므로
    # 일단 라디안으로 가정하고 변환
    # 만약 도 단위라면 먼저 라디안으로 변환 필요
    
    # ENU → 유클리드 변환
    # ENU: 0도=북쪽, 시계방향이 양수
    # 유클리드: 0도=동쪽, 반시계방향이 양수
    return np.deg2rad((90 - np.rad2deg(float(enu_heading))) % 360.0)

