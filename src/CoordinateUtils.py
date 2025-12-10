"""
좌표 변환 유틸리티 모듈

이 모듈은 다양한 좌표계 간 변환을 수행하는 유틸리티 함수들을 제공합니다.
- 로컬 좌표 → UTM 좌표 변환
- 월드 좌표 → Ego 중심 좌표계 변환
- 다양한 포인트 타입을 (x, y) 튜플로 변환

참조:
    - doc/01_architecture.md: 좌표 변환 설명
"""
from __future__ import annotations
# pylint: disable=unsubscriptable-object

from typing import Tuple, Union, List
import numpy as np


def local_to_utm(local_x: float, local_y: float, map_origin_x: float, map_origin_y: float) -> Tuple[float, float]:
    """
    로컬 좌표를 UTM 좌표로 변환합니다.
    
    JSON 로그 파일의 로컬 좌표계(상대 좌표)를 UTM 좌표계로 변환합니다.
    맵 원점(MAP_ORIGIN_X, MAP_ORIGIN_Y)을 더하여 변환합니다.
    
    Args:
        local_x (float): 로컬 X 좌표 (미터)
        local_y (float): 로컬 Y 좌표 (미터)
        map_origin_x (float): 맵 원점 X 좌표 (UTM, 미터)
        map_origin_y (float): 맵 원점 Y 좌표 (UTM, 미터)
    
    Returns:
        Tuple[float, float]: UTM 좌표 (x, y)
    
    변환 공식:
        UTM_x = local_x + map_origin_x
        UTM_y = local_y + map_origin_y
    
    사용 예시:
        utm_x, utm_y = local_to_utm(10.5, 20.3, 230388.61912, 424695.37128)
        # 결과: (230399.11912, 424715.67128)
    
    참조:
        - JsonLogLoader: 로컬 좌표를 UTM 좌표로 변환
    """
    return float(local_x + map_origin_x), float(local_y + map_origin_y)


def world_to_ego_centric(points: np.ndarray, ego_origin: np.ndarray, ego_angle: float) -> np.ndarray:
    """
    월드 좌표를 Ego 중심 좌표로 변환합니다.
    
    Ego 차량의 위치와 헤딩을 기준으로 좌표를 변환합니다.
    변환 후 Ego 차량은 원점 (0, 0)에 위치하고 헤딩은 0도가 됩니다.
    
    Args:
        points (np.ndarray): 변환할 점들
            형태: (N, 2) 또는 (2,) - (x, y) 좌표
        ego_origin (np.ndarray): Ego 차량의 원점 좌표
            형태: (2,) 또는 (3,) - (x, y) 또는 (x, y, heading)
            (3,)인 경우 처음 두 요소만 사용됩니다.
        ego_angle (float): Ego 차량의 헤딩 각도 (라디안)
    
    Returns:
        np.ndarray: 변환된 좌표 (Ego 중심 좌표계)
            Ego 차량이 원점 (0, 0)에 위치하고 전방이 +X 방향
    
    변환 과정:
        1. 평행이동: points - ego_origin[:2]
        2. 회전: rot_mat @ (points - ego_origin[:2])
    
    회전 행렬:
        rot_mat = [[cos(angle), -sin(angle)],
                   [sin(angle),  cos(angle)]]
    
    사용 예시:
        points = np.array([[10.0, 20.0], [15.0, 25.0]])
        ego_origin = np.array([100.0, 200.0, 0.5])  # (x, y, heading)
        ego_angle = 0.5  # 라디안
        transformed = world_to_ego_centric(points, ego_origin, ego_angle)
    
    참조:
        - CustomScenarioVisualizer: 시각화를 위한 좌표 변환
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Ego 원점 추출 (처음 두 요소만 사용)
    origin_xy = ego_origin[:2] if len(ego_origin) >= 2 else ego_origin
    
    # 회전 행렬 생성
    rot_mat = np.array([
        [np.cos(ego_angle), -np.sin(ego_angle)],
        [np.sin(ego_angle), np.cos(ego_angle)]
    ], dtype=np.float64)
    
    # 평행이동 후 회전
    return np.matmul(points - origin_xy, rot_mat)


def coerce_xy(point_like: Union[Tuple[float, float], List[float], object]) -> Tuple[float, float]:
    """
    다양한 포인트 타입을 (x, y) 튜플로 변환합니다.
    
    다양한 형태의 포인트 입력을 받아서 (x, y) 튜플로 변환합니다.
    다음 형태들을 지원합니다:
    - tuple/list of length 2: (x, y) 또는 [x, y]
    - objects with attributes .x and .y
    - objects with attribute .array (first two as x, y)
    - objects with nested .center or .point exposing .x/.y
    
    Args:
        point_like: 변환할 포인트
            지원되는 형태:
            - Tuple[float, float] 또는 List[float]: (x, y) 또는 [x, y]
            - 객체: .x, .y 속성을 가진 객체
            - 객체: .array 속성을 가진 객체 (첫 두 요소가 x, y)
            - 객체: .center 또는 .point 속성을 가진 객체
    
    Returns:
        Tuple[float, float]: (x, y) 좌표 튜플
    
    Raises:
        TypeError: 지원되지 않는 포인트 타입인 경우
    
    사용 예시:
        # 튜플/리스트
        x, y = coerce_xy((10.0, 20.0))
        x, y = coerce_xy([10.0, 20.0])
        
        # 객체 (x, y 속성)
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        x, y = coerce_xy(Point(10.0, 20.0))
        
        # 객체 (array 속성)
        class PointArray:
            def __init__(self, arr):
                self.array = arr
        x, y = coerce_xy(PointArray(np.array([10.0, 20.0, 30.0])))
    
    참조:
        - MapManager: 맵 쿼리에서 포인트 변환
    """
    # tuple/list
    if isinstance(point_like, (tuple, list)) and len(point_like) >= 2:
        return float(point_like[0]), float(point_like[1])
    
    # numpy-like array
    arr = getattr(point_like, "array", None)
    if arr is not None:
        try:
            return float(arr[0]), float(arr[1])
        except (IndexError, TypeError, ValueError):
            pass
    
    # object with x, y
    x = getattr(point_like, "x", None)
    y = getattr(point_like, "y", None)
    if x is not None and y is not None:
        return float(x), float(y)
    
    # nested center/point
    center = getattr(point_like, "center", None)
    if center is not None:
        cx = getattr(center, "x", None)
        cy = getattr(center, "y", None)
        if cx is not None and cy is not None:
            return float(cx), float(cy)
        point = getattr(center, "point", None)
        if point is not None:
            px = getattr(point, "x", None)
            py = getattr(point, "y", None)
            if px is not None and py is not None:
                return float(px), float(py)
    
    point = getattr(point_like, "point", None)
    if point is not None:
        px = getattr(point, "x", None)
        py = getattr(point, "y", None)
        if px is not None and py is not None:
            return float(px), float(py)
    
    # fallback
    raise TypeError(f"Unsupported point-like input for XY coercion: {type(point_like)}")

