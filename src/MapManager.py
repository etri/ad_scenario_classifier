"""SQLite 기반 맵 데이터베이스 관리 및 공간 쿼리 모듈."""

from __future__ import annotations
# pylint: disable=unsubscriptable-object

import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

from shapely import wkb
from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import numpy as np

# 유틸리티 모듈
from CoordinateUtils import coerce_xy
from FileUtils import validate_file_path, get_absolute_path


LAYER_LANE = "LANE"
LAYER_LANE_CONNECTOR = "LANE_CONNECTOR"
LAYER_ROADBLOCK = "ROADBLOCK"
LAYER_ROADBLOCK_CONNECTOR = "ROADBLOCK_CONNECTOR"
LAYER_STOP_LINE = "STOP_LINE"
LAYER_CROSSWALK = "CROSSWALK"
LAYER_CARPARK_AREA = "CARPARK_AREA"

LayerLike = Union[str, object]


@dataclass(eq=False)
class PolylineMapObject:
    """NuPlan 호환 폴리라인 맵 객체."""
    geometry: BaseGeometry
    distance_for_heading_estimation: float = 0.5

    @property
    def linestring(self) -> BaseGeometry:
        return self.geometry

    @property
    def length(self) -> float:
        return float(self.geometry.length) if self.geometry is not None else 0.0

    @property
    def discrete_path(self) -> List['StateSE2']:
        coords = list(self.geometry.coords) if (self.geometry is not None and not self.geometry.is_empty) else []
        if len(coords) == 0:
            return []
        states: List[StateSE2] = []
        for i, (x, y) in enumerate(coords):
            if i == 0:
                nx, ny = coords[1] if len(coords) > 1 else (x, y)
                dx, dy = nx - x, ny - y
            else:
                px, py = coords[i - 1]
                dx, dy = x - px, y - py
            hd = math.atan2(dy, dx) if (dx != 0.0 or dy != 0.0) else 0.0
            states.append(StateSE2(float(x), float(y), float(hd)))
        return states

    def get_nearest_arc_length_from_position(self, point_xy: Tuple[float, float]) -> float:
        return float(self.geometry.project(Point(point_xy[0], point_xy[1])))

    def get_nearest_pose_from_position(self, point_xy: Tuple[float, float]) -> StateSE2:
        s = self.get_nearest_arc_length_from_position(point_xy)
        p1 = self.geometry.interpolate(s)
        p2 = self.geometry.interpolate(s + self.distance_for_heading_estimation)
        if p1.equals(p2):
            p2 = self.geometry.interpolate(max(0.0, s - self.distance_for_heading_estimation))
        hd = math.atan2(p2.y - p1.y, p2.x - p1.x)
        return StateSE2(float(p1.x), float(p1.y), float(hd))

    def __getattr__(self, name: str):
        return getattr(self.geometry, name)

    def __hash__(self) -> int:
        return hash(id(self.geometry))

def _load_geom(blob: Optional[bytes]) -> Optional[BaseGeometry]:
    if blob is None:
        return None
    try:
        return wkb.loads(blob)
    except (ValueError, TypeError, RuntimeError):
        return None


def _to_layer_name(layer: LayerLike) -> str:
    if hasattr(layer, "name"):
        return str(getattr(layer, "name")).upper()
    return str(layer).upper()


def _query(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _q(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    return _query(conn, sql, params)
@dataclass
class StateSE2:
    x: float
    y: float
    heading: float

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading], dtype=np.float64)


@dataclass
class Point2D:
    x: float
    y: float

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


def _state_point(self: StateSE2) -> Point2D:
    return Point2D(self.x, self.y)

setattr(StateSE2, "point", property(_state_point))


@dataclass(eq=False)
class Path:
    discrete_path: List[StateSE2]

    @property
    def length(self) -> float:
        if len(self.discrete_path) < 2:
            return 0.0
        arr = np.array([[s.x, s.y] for s in self.discrete_path], dtype=np.float64)
        dif = np.diff(arr, axis=0)
        seg = np.hypot(dif[:, 0], dif[:, 1])
        return float(seg.sum())

    @property
    def linestring(self) -> LineString:
        try:
            if not self.discrete_path:
                return LineString([])
            coords = [(s.x, s.y) for s in self.discrete_path]
            return LineString(coords)
        except (ValueError, TypeError, AttributeError):
            return LineString([])


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    ok = cur.fetchone() is not None
    cur.close()
    return ok


def _has_columns(conn: sqlite3.Connection, table: str, columns: Iterable[str]) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    names = {row[1] for row in cur.fetchall()}
    cur.close()
    return all(col in names for col in columns)


@dataclass(eq=False)
class Lane:
    id: str
    roadblock_id: Optional[str]
    polygon: BaseGeometry
    baseline: BaseGeometry
    baseline_sampled: Optional[BaseGeometry]
    left_boundary: BaseGeometry
    right_boundary: BaseGeometry
    length_m: float
    speed_limit_mps: Optional[float]
    width_m: Optional[float]
    baseline_path: Optional['Path'] = None
    
    # New fields from LANE_LINK
    is_bidirectional: Optional[bool] = None
    start_node_id: Optional[int] = None
    end_node_id: Optional[int] = None
    junction: Optional[int] = None
    lane_type: Optional[int] = None
    sub_type: Optional[int] = None
    vehicle_traffic_light_id: Optional[int] = None
    left_link_id: Optional[int] = None
    right_link_id: Optional[int] = None

    # Graph edges (wired post-load)
    outgoing_edges: List['Lane'] = field(default_factory=list)
    incoming_edges: List['Lane'] = field(default_factory=list)

    def get_roadblock_id(self) -> Optional[str]:
        return self.roadblock_id

    def contains_point(self, point: Union[Point, Tuple[float, float]]) -> bool:
        try:
            if isinstance(point, Point):
                p = point
            else:
                x = y = None
                if isinstance(point, (tuple, list)) and len(point) >= 2:
                    x, y = float(point[0]), float(point[1])
                else:
                    # object with x, y
                    px = getattr(point, "x", None)
                    py = getattr(point, "y", None)
                    if px is not None and py is not None:
                        x, y = float(px), float(py)
                    else:
                        arr = getattr(point, "array", None)
                        if arr is not None:
                            try:
                                x, y = float(arr[0]), float(arr[1])
                            except (IndexError, TypeError, ValueError):
                                x = y = None
                        if x is None or y is None:
                            center = getattr(point, "center", None)
                            if center is not None:
                                cx = getattr(center, "x", None)
                                cy = getattr(center, "y", None)
                                if cx is not None and cy is not None:
                                    x, y = float(cx), float(cy)
                                else:
                                    cp = getattr(center, "point", None)
                                    if cp is not None:
                                        cpx = getattr(cp, "x", None)
                                        cpy = getattr(cp, "y", None)
                                        if cpx is not None and cpy is not None:
                                            x, y = float(cpx), float(cpy)
                            if (x is None or y is None) and getattr(point, "point", None) is not None:
                                pt = getattr(point, "point")
                                ptx = getattr(pt, "x", None)
                                pty = getattr(pt, "y", None)
                                if ptx is not None and pty is not None:
                                    x, y = float(ptx), float(pty)
                if x is None or y is None:
                    return False
                p = Point(x, y)
            return bool(self.polygon.contains(p))
        except (TypeError, AttributeError, ValueError):
            return False


@dataclass(eq=False)
class LaneConnector:
    id: str
    roadblock_connector_id: Optional[str]
    polygon: BaseGeometry
    baseline: BaseGeometry
    length_m: float
    baseline_sampled: Optional[BaseGeometry] = None
    from_lane_id: Optional[str] = None
    to_lane_id: Optional[str] = None
    speed_limit_mps: Optional[float] = None
    width_m: Optional[float] = None
    left_boundary: Optional[BaseGeometry] = None
    right_boundary: Optional[BaseGeometry] = None
    baseline_path: Optional['Path'] = None

    def get_roadblock_id(self) -> Optional[str]:
        return self.roadblock_connector_id


@dataclass
class Roadblock:
    id: str
    polygon: BaseGeometry
    interior_edges: List[str]
    interior_edge_objs: List['Lane'] = field(default_factory=list)
    incoming_edges: List['Roadblock'] = field(default_factory=list)
    outgoing_edges: List['Roadblock'] = field(default_factory=list)


@dataclass
class RoadblockConnector:
    id: str
    polygon: BaseGeometry
    interior_edges: List[str]
    interior_edge_objs: List['LaneConnector'] = field(default_factory=list)
    incoming_edges: List['RoadblockConnector'] = field(default_factory=list)
    outgoing_edges: List['RoadblockConnector'] = field(default_factory=list)


@dataclass
class StopLine:
    id: str
    linestring: BaseGeometry


@dataclass
class Crosswalk:
    id: str
    polygon: BaseGeometry


@dataclass
class CarparkArea:
    id: str
    polygon: BaseGeometry


@dataclass
class Laneside:
    id: int
    mid: int
    lane_type: int
    color: int
    num_points: int
    linestring: BaseGeometry


@dataclass 
class RoadLight:
    id: int
    lane_id: int
    light_type: int
    sub_type: int
    div: int
    uturn: int
    num_stop_lines: int
    stop_line_ids: List[int]
    num_points: int
    linestring: BaseGeometry


@dataclass
class PedestrianLight:
    id: int
    num_crosswalks: int
    crosswalk_ids: List[int]
    light_type: int
    direction: float
    x: float
    y: float


class MapManager:
    """SQLite 맵 데이터베이스 관리 및 STRtree 기반 공간 쿼리 제공."""

    def __init__(self, sqlite_path: str) -> None:
        """
        Args:
            sqlite_path: SQLite 맵 데이터베이스 파일 경로
        """
        # 파일 경로 검증
        validate_file_path(sqlite_path, "SQLite database file")
        self.sqlite_path = sqlite_path  # SQLite 파일 경로 저장

        self._lanes: Dict[str, Lane] = {}
        self._lane_connectors: Dict[str, LaneConnector] = {}
        self._roadblocks: Dict[str, Roadblock] = {}
        self._roadblock_connectors: Dict[str, RoadblockConnector] = {}
        self._stop_lines: Dict[str, StopLine] = {}
        self._crosswalks: Dict[str, Crosswalk] = {}
        self._lanesides: Dict[int, Laneside] = {}
        self._roadlights: Dict[int, RoadLight] = {}
        self._pedestrian_lights: Dict[int, PedestrianLight] = {}

        self._lane_tree: Optional[STRtree] = None
        self._lane_geoms: List[BaseGeometry] = []
        self._lane_ids: List[str] = []
        self._lane_gid_map: Dict[int, str] = {}

        self._lc_tree: Optional[STRtree] = None
        self._lc_geoms: List[BaseGeometry] = []
        self._lc_ids: List[str] = []
        self._lc_gid_map: Dict[int, str] = {}

        self._rb_tree: Optional[STRtree] = None
        self._rb_geoms: List[BaseGeometry] = []
        self._rb_ids: List[str] = []
        self._rb_gid_map: Dict[int, str] = {}

        self._rbc_tree: Optional[STRtree] = None
        self._rbc_geoms: List[BaseGeometry] = []
        self._rbc_ids: List[str] = []
        self._rbc_gid_map: Dict[int, str] = {}

        self._sl_tree: Optional[STRtree] = None
        self._sl_geoms: List[BaseGeometry] = []
        self._sl_ids: List[str] = []
        self._sl_gid_map: Dict[int, str] = {}

        self._cw_tree: Optional[STRtree] = None
        self._cw_geoms: List[BaseGeometry] = []
        self._cw_ids: List[str] = []
        self._cw_gid_map: Dict[int, str] = {}

        self._carpark_store: Dict[str, BaseGeometry] = {}
        self._carpark_objects: Dict[str, CarparkArea] = {}
        self._carpark_tree: Optional[STRtree] = None
        self._carpark_geoms: List[BaseGeometry] = []
        self._carpark_ids: List[str] = []
        self._carpark_gid_map: Dict[int, str] = {}

        self._lane_successors: Dict[str, List[str]] = {}
        self._lane_predecessors: Dict[str, List[str]] = {}
        self._from_lane_to_connectors: Dict[str, List[str]] = {}
        self._pair_to_connectors: Dict[Tuple[str, str], List[str]] = {}

    def initialize_all_layers(self) -> None:
        """모든 맵 레이어를 로드하고 STRtree 공간 인덱스 구축."""
        try:
            conn = sqlite3.connect(self.sqlite_path)
        except sqlite3.OperationalError as e:
            abs_path = get_absolute_path(self.sqlite_path)
            raise sqlite3.OperationalError(
                f"Failed to connect to SQLite database: {self.sqlite_path}\n"
                f"   Absolute path: {abs_path}\n"
                f"   This may be due to:\n"
                f"   - Database file is locked by another process\n"
                f"   - Database file is corrupted\n"
                f"   - Insufficient permissions to access the file\n"
                f"   Original error: {e}"
            ) from e
        except sqlite3.Error as e:
            abs_path = get_absolute_path(self.sqlite_path)
            raise sqlite3.Error(
                f"SQLite error when connecting to database: {self.sqlite_path}\n"
                f"   Absolute path: {abs_path}\n"
                f"   Details: {e}"
            ) from e
        
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

        self._wire_graphs()
        if self._lanes:
            self._lane_geoms, self._lane_ids, self._lane_gid_map = [], [], {}
            for lid, obj in self._lanes.items():
                g = obj.polygon
                if g is None or g.is_empty:
                    continue
                self._lane_geoms.append(g)
                self._lane_ids.append(lid)
                self._lane_gid_map[id(g)] = lid
            self._lane_tree = STRtree(self._lane_geoms)
        if self._lane_connectors:
            self._lc_geoms, self._lc_ids, self._lc_gid_map = [], [], {}
            for cid, obj in self._lane_connectors.items():
                g = obj.polygon
                if g is None or g.is_empty:
                    continue
                self._lc_geoms.append(g)
                self._lc_ids.append(cid)
                self._lc_gid_map[id(g)] = cid
            self._lc_tree = STRtree(self._lc_geoms)
        if self._roadblocks:
            self._rb_geoms, self._rb_ids, self._rb_gid_map = [], [], {}
            for rid, obj in self._roadblocks.items():
                g = obj.polygon
                if g is None or g.is_empty:
                    continue
                self._rb_geoms.append(g)
                self._rb_ids.append(rid)
                self._rb_gid_map[id(g)] = rid
            self._rb_tree = STRtree(self._rb_geoms)
        if self._roadblock_connectors:
            self._rbc_geoms, self._rbc_ids, self._rbc_gid_map = [], [], {}
            for rid, obj in self._roadblock_connectors.items():
                g = obj.polygon
                if g is None or g.is_empty:
                    continue
                self._rbc_geoms.append(g)
                self._rbc_ids.append(rid)
                self._rbc_gid_map[id(g)] = rid
            self._rbc_tree = STRtree(self._rbc_geoms)
        if self._stop_lines:
            self._sl_geoms, self._sl_ids, self._sl_gid_map = [], [], {}
            for sid, obj in self._stop_lines.items():
                g = obj.linestring
                if g is None or g.is_empty:
                    continue
                self._sl_geoms.append(g)
                self._sl_ids.append(sid)
                self._sl_gid_map[id(g)] = sid
            self._sl_tree = STRtree(self._sl_geoms)
        if self._crosswalks:
            self._cw_geoms, self._cw_ids, self._cw_gid_map = [], [], {}
            for cid, obj in self._crosswalks.items():
                g = obj.polygon
                if g is None or g.is_empty:
                    continue
                self._cw_geoms.append(g)
                self._cw_ids.append(cid)
                self._cw_gid_map[id(g)] = cid
            self._cw_tree = STRtree(self._cw_geoms)

        # Carpark STRtree
        if self._carpark_store:
            self._carpark_geoms, self._carpark_ids, self._carpark_gid_map = [], [], {}
            for cid, geom in self._carpark_store.items():
                if geom is None or geom.is_empty:
                    continue
                self._carpark_geoms.append(geom)
                self._carpark_ids.append(cid)
                self._carpark_gid_map[id(geom)] = cid
            self._carpark_tree = STRtree(self._carpark_geoms)

    def _load_lanes(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "lanes"):
            return
        has_speed = _has_columns(conn, "lanes", ["speed_limit_mps"])
        has_sampled = _has_columns(conn, "lanes", ["baseline_sampled_wkb"])
        has_width = _has_columns(conn, "lanes", ["width_m"])
        has_new_fields = _has_columns(conn, "lanes", [
            "is_bidirectional", "start_node_id", "end_node_id", "junction", 
            "lane_type", "sub_type", "vehicle_traffic_light_id", "left_link_id", "right_link_id"
        ])

        # Base fields
        base_select = "id, roadblock_id, speed_limit_mps, polygon_wkb, baseline_wkb, left_bound_wkb, right_bound_wkb"
        
        # Optional fields
        sampled_field = "baseline_sampled_wkb" if has_sampled else "NULL as baseline_sampled_wkb"
        width_field = "width_m" if has_width else "NULL as width_m"
        
        # New fields
        if has_new_fields:
            new_fields = "is_bidirectional, start_node_id, end_node_id, junction, lane_type, sub_type, vehicle_traffic_light_id, left_link_id, right_link_id"
        else:
            new_fields = "NULL as is_bidirectional, NULL as start_node_id, NULL as end_node_id, NULL as junction, NULL as lane_type, NULL as sub_type, NULL as vehicle_traffic_light_id, NULL as left_link_id, NULL as right_link_id"
            
        sql = f"SELECT {base_select}, {sampled_field}, {width_field}, length_m, {new_fields} FROM lanes"

        rows = _query(conn, sql)
        for row in rows:
            (lid, rb_id, spd, poly_wkb, bl_wkb, l_wkb, r_wkb, bls_wkb, width_m, length_m,
             is_bidirectional, start_node_id, end_node_id, junction, lane_type, sub_type, 
             vehicle_traffic_light_id, left_link_id, right_link_id) = row
            poly = _load_geom(poly_wkb)
            baseline = _load_geom(bl_wkb)
            baseline_smp = _load_geom(bls_wkb)
            left = _load_geom(l_wkb)
            right = _load_geom(r_wkb)
            # Build paths and wrappers
            base_or_poly = baseline or (poly.boundary if poly is not None else None)
            baseline_path = self._line_to_path(base_or_poly)
            left_geom = left or base_or_poly
            right_geom = right or base_or_poly
            left_path = self._line_to_path(left_geom)
            right_path = self._line_to_path(right_geom)
            self._lanes[str(lid)] = Lane(
                id=str(lid),
                roadblock_id=str(rb_id) if rb_id is not None else None,
                polygon=poly,
                baseline=base_or_poly,
                baseline_sampled=baseline_smp,
                baseline_path=baseline_path,
                left_boundary=PolylineMapObject(geometry=left_geom),
                right_boundary=PolylineMapObject(geometry=right_geom),
                length_m=float(length_m) if length_m is not None else (baseline.length if baseline is not None else 0.0),
                speed_limit_mps=float(spd) if (has_speed and spd is not None) else None,
                width_m=float(width_m) if (has_width and width_m is not None) else None,
                # New fields
                is_bidirectional=bool(is_bidirectional) if is_bidirectional is not None else None,
                start_node_id=int(start_node_id) if start_node_id is not None else None,
                end_node_id=int(end_node_id) if end_node_id is not None else None,
                junction=int(junction) if junction is not None else None,
                lane_type=int(lane_type) if lane_type is not None else None,
                sub_type=int(sub_type) if sub_type is not None else None,
                vehicle_traffic_light_id=int(vehicle_traffic_light_id) if vehicle_traffic_light_id is not None else None,
                left_link_id=int(left_link_id) if left_link_id is not None else None,
                right_link_id=int(right_link_id) if right_link_id is not None else None,
            )

    def _load_lane_connectors(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "lane_connectors"):
            return
        has_speed = _has_columns(conn, "lane_connectors", ["speed_limit_mps"])
        has_sampled = _has_columns(conn, "lane_connectors", ["baseline_sampled_wkb"])
        has_width = _has_columns(conn, "lane_connectors", ["width_m"])
        has_bounds = _has_columns(conn, "lane_connectors", ["left_bound_wkb", "right_bound_wkb"])

        if has_bounds and has_sampled and has_width:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, left_bound_wkb, right_bound_wkb, "
                "baseline_sampled_wkb, width_m, length_m FROM lane_connectors"
            )
        elif has_bounds and has_sampled:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, left_bound_wkb, right_bound_wkb, "
                "baseline_sampled_wkb, NULL as width_m, length_m FROM lane_connectors"
            )
        elif has_bounds and has_width:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, left_bound_wkb, right_bound_wkb, "
                "NULL as baseline_sampled_wkb, width_m, length_m FROM lane_connectors"
            )
        elif has_bounds:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, left_bound_wkb, right_bound_wkb, "
                "NULL as baseline_sampled_wkb, NULL as width_m, length_m FROM lane_connectors"
            )
        elif has_sampled and has_width:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, NULL as left_bound_wkb, NULL as right_bound_wkb, "
                "baseline_sampled_wkb, width_m, length_m FROM lane_connectors"
            )
        elif has_sampled:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, NULL as left_bound_wkb, NULL as right_bound_wkb, "
                "baseline_sampled_wkb, NULL as width_m, length_m FROM lane_connectors"
            )
        elif has_width:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, NULL as left_bound_wkb, NULL as right_bound_wkb, "
                "NULL as baseline_sampled_wkb, width_m, length_m FROM lane_connectors"
            )
        else:
            sql = (
                "SELECT id, roadblock_connector_id, from_lane_id, to_lane_id, speed_limit_mps, polygon_wkb, baseline_wkb, NULL as left_bound_wkb, NULL as right_bound_wkb, "
                "NULL as baseline_sampled_wkb, NULL as width_m, length_m FROM lane_connectors"
            )

        rows = _query(conn, sql)
        for row in rows:
            (cid, rbc_id, from_id, to_id, spd, poly_wkb, bl_wkb, l_wkb, r_wkb, bls_wkb, width_m, length_m) = row
            poly = _load_geom(poly_wkb)
            baseline = _load_geom(bl_wkb)
            left = _load_geom(l_wkb)
            right = _load_geom(r_wkb)
            baseline_smp = _load_geom(bls_wkb)
            base_or_poly = baseline or (poly.boundary if poly is not None else None)
            baseline_path = self._line_to_path(base_or_poly)
            left_geom = left or base_or_poly
            right_geom = right or base_or_poly
            left_path = self._line_to_path(left_geom)
            right_path = self._line_to_path(right_geom)
            self._lane_connectors[str(cid)] = LaneConnector(
                id=str(cid),
                roadblock_connector_id=str(rbc_id) if rbc_id is not None else None,
                polygon=poly,
                baseline=base_or_poly,
                baseline_sampled=baseline_smp,
            left_boundary=PolylineMapObject(geometry=left_geom) if left_geom is not None else None,
            right_boundary=PolylineMapObject(geometry=right_geom) if right_geom is not None else None,
                baseline_path=baseline_path,
                from_lane_id=str(from_id) if from_id is not None else None,
                to_lane_id=str(to_id) if to_id is not None else None,
                length_m=float(length_m) if length_m is not None else (baseline.length if baseline is not None else 0.0),
                speed_limit_mps=float(spd) if (has_speed and spd is not None) else None,
                width_m=float(width_m) if (has_width and width_m is not None) else None,
            )

            if from_id is not None:
                self._from_lane_to_connectors.setdefault(str(from_id), []).append(str(cid))
            if from_id is not None and to_id is not None:
                self._pair_to_connectors.setdefault((str(from_id), str(to_id)), []).append(str(cid))

    def _load_roadblocks(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "roadblocks"):
            return
        rows = _query(conn, "SELECT id, polygon_wkb FROM roadblocks")
        rb_map: Dict[str, BaseGeometry] = {str(_id): _load_geom(poly) for _id, poly in rows}
        interior: Dict[str, List[str]] = {}
        if _table_exists(conn, "roadblock_interior_edges"):
            for rb_id, lane_id in _query(conn, "SELECT roadblock_id, lane_id FROM roadblock_interior_edges"):
                interior.setdefault(str(rb_id), []).append(str(lane_id))
        for _id, poly in rb_map.items():
            lane_objs: List[Lane] = []
            for lid in interior.get(_id, []):
                ln = self._lanes.get(str(lid))
                if ln is not None:
                    lane_objs.append(ln)
            self._roadblocks[_id] = Roadblock(id=_id, polygon=poly, interior_edges=lane_objs)

    def _load_roadblock_connectors(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "roadblock_connectors"):
            return
        rows = _query(conn, "SELECT id, polygon_wkb FROM roadblock_connectors")
        rbc_map: Dict[str, BaseGeometry] = {str(_id): _load_geom(poly) for _id, poly in rows}
        interior: Dict[str, List[str]] = {}
        if _table_exists(conn, "rbc_interior_edges"):
            for rbc_id, lc_id in _query(conn, "SELECT roadblock_connector_id, lane_connector_id FROM rbc_interior_edges"):
                interior.setdefault(str(rbc_id), []).append(str(lc_id))
        for _id, poly in rbc_map.items():
            conn_objs: List[LaneConnector] = []
            for cid in interior.get(_id, []):
                lc = self._lane_connectors.get(str(cid))
                if lc is not None:
                    conn_objs.append(lc)
            self._roadblock_connectors[_id] = RoadblockConnector(id=_id, polygon=poly, interior_edges=conn_objs)

    def _load_lane_graph(self, conn: sqlite3.Connection) -> None:
        self._lane_successors.clear()
        self._lane_predecessors.clear()
        if not _table_exists(conn, "lane_successors"):
            return
        for from_id, to_id in _query(conn, "SELECT from_lane_id, to_lane_id FROM lane_successors"):
            f = str(from_id)
            t = str(to_id)
            self._lane_successors.setdefault(f, []).append(t)
            self._lane_predecessors.setdefault(t, []).append(f)

    def _load_stop_lines(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "stop_lines"):
            return
        for _id, line_wkb in _query(conn, "SELECT id, linestring_wkb FROM stop_lines"):
            self._stop_lines[str(_id)] = StopLine(id=str(_id), linestring=_load_geom(line_wkb))

    def _load_crosswalks(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "crosswalks"):
            return
        for _id, poly_wkb in _query(conn, "SELECT id, polygon_wkb FROM crosswalks"):
            self._crosswalks[str(_id)] = Crosswalk(id=str(_id), polygon=_load_geom(poly_wkb))

    def _load_carpark_areas(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "carpark_areas"):
            return
        rows = _query(conn, "SELECT id, polygon_wkb FROM carpark_areas")
        for _id, blob in rows:
            geom = _load_geom(blob)
            if geom is None or geom.is_empty:
                continue
            sid = str(_id)
            self._carpark_store[sid] = geom
            self._carpark_objects[sid] = CarparkArea(id=sid, polygon=geom)

    def _load_lanesides(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "lanesides"):
            return
        rows = _query(conn, "SELECT id, mid, type, color, num_points, linestring_wkb FROM lanesides")
        for ls_id, mid, ls_type, color, num_points, line_wkb in rows:
            linestring = _load_geom(line_wkb)
            if linestring is None:
                continue
            self._lanesides[int(ls_id)] = Laneside(
                id=int(ls_id),
                mid=int(mid),
                lane_type=int(ls_type),
                color=int(color),
                num_points=int(num_points),
                linestring=linestring
            )

    def _load_roadlights(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "roadlights"):
            return
        rows = _query(conn, "SELECT id, lane_id, type, sub_type, div, uturn, num_stop_lines, stop_line_ids, num_points, linestring_wkb FROM roadlights")
        for rl_id, lane_id, rl_type, sub_type, div, uturn, num_stop_lines, stop_line_ids_str, num_points, line_wkb in rows:
            linestring = _load_geom(line_wkb)
            if linestring is None:
                continue
            
            stop_line_ids = []
            if stop_line_ids_str:
                try:
                    stop_line_ids = [int(x.strip()) for x in stop_line_ids_str.split(",") if x.strip()]
                except ValueError:
                    stop_line_ids = []
                    
            self._roadlights[int(rl_id)] = RoadLight(
                id=int(rl_id),
                lane_id=int(lane_id),
                light_type=int(rl_type),
                sub_type=int(sub_type),
                div=int(div),
                uturn=int(uturn),
                num_stop_lines=int(num_stop_lines),
                stop_line_ids=stop_line_ids,
                num_points=int(num_points),
                linestring=linestring
            )

    def _load_pedestrian_lights(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "pedestrian_lights"):
            return
        rows = _query(conn, "SELECT id, num_crosswalks, crosswalk_ids, type, direction, x, y FROM pedestrian_lights")
        for pl_id, num_crosswalks, crosswalk_ids_str, pl_type, direction, x, y in rows:
            crosswalk_ids = []
            if crosswalk_ids_str:
                try:
                    crosswalk_ids = [int(x.strip()) for x in crosswalk_ids_str.split(",") if x.strip()]
                except ValueError:
                    crosswalk_ids = []
                    
            self._pedestrian_lights[int(pl_id)] = PedestrianLight(
                id=int(pl_id),
                num_crosswalks=int(num_crosswalks),
                crosswalk_ids=crosswalk_ids,
                light_type=int(pl_type),
                direction=float(direction) if direction is not None else 0.0,
                x=float(x),
                y=float(y)
            )

    def _coerce_xy(self, point_like: Union[Tuple[float, float], List[float], object]) -> Tuple[float, float]:
        """
        Coerce various point-like inputs to (x, y):
        - tuple/list of length 2
        - objects with attributes .x and .y
        - objects with attribute .array (first two as x, y)
        - objects with nested .center or .point exposing .x/.y
        
        이 메서드는 CoordinateUtils.coerce_xy()를 호출합니다.
        """
        return coerce_xy(point_like)

    def get_proximal_map_objects(self, point, radius: float, layers: Iterable[LayerLike]) -> Dict[LayerLike, List[object]]:
        """
        Args:
            point: 쿼리 기준점 (다양한 포인트 형식 지원)
            radius: 검색 반경 (미터)
            layers: 조회할 맵 레이어 리스트
        
        Returns:
            레이어별 맵 객체 리스트 딕셔너리
        """
        try:
            px, py = coerce_xy(point)
        except (TypeError, ValueError, AttributeError):
            px, py = point
        p = Point(px, py)
        query_geom = p.buffer(radius).envelope

        results: Dict[LayerLike, List[object]] = {}
        for layer in layers:
            lname = _to_layer_name(layer)
            objs: List[object] = []
            if lname == LAYER_LANE:
                if self._lane_tree is not None:
                    for item in self._lane_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            lid = self._lane_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                lid = self._lane_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                lid = None
                        if lid is None:
                            continue
                        objs.append(self._lanes[lid])
            elif lname == LAYER_LANE_CONNECTOR:
                if self._lc_tree is not None:
                    for item in self._lc_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            cid = self._lc_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                cid = self._lc_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                cid = None
                        if cid is None:
                            continue
                        objs.append(self._lane_connectors[cid])
            elif lname == LAYER_ROADBLOCK:
                if self._rb_tree is not None:
                    for item in self._rb_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            rid = self._rb_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                rid = self._rb_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                rid = None
                        if rid is None:
                            continue
                        objs.append(self._roadblocks[rid])
            elif lname == LAYER_ROADBLOCK_CONNECTOR:
                if self._rbc_tree is not None:
                    for item in self._rbc_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            rid = self._rbc_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                rid = self._rbc_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                rid = None
                        if rid is None:
                            continue
                        objs.append(self._roadblock_connectors[rid])
            elif lname == LAYER_STOP_LINE:
                if self._sl_tree is not None:
                    for item in self._sl_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            sid = self._sl_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                sid = self._sl_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                sid = None
                        if sid is None:
                            continue
                        objs.append(self._stop_lines[sid])
            elif lname == LAYER_CROSSWALK:
                if self._cw_tree is not None:
                    for item in self._cw_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            cid = self._cw_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                cid = self._cw_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                cid = None
                        if cid is None:
                            continue
                        objs.append(self._crosswalks[cid])
            elif lname == LAYER_CARPARK_AREA:
                if self._carpark_tree is not None:
                    for item in self._carpark_tree.query(query_geom):
                        if isinstance(item, BaseGeometry):
                            cid = self._carpark_gid_map.get(id(item))
                        else:
                            try:
                                idx = int(item)
                                cid = self._carpark_ids[idx]
                            except (ValueError, TypeError, IndexError):
                                cid = None
                        if cid is None:
                            continue
                        objs.append(self._carpark_objects[cid])
            results[layer] = objs
        return results

    def get_map_object(self, object_id: Union[str, int], layer: LayerLike) -> Optional[object]:
        oid = str(object_id)
        lname = _to_layer_name(layer)
        if lname == LAYER_LANE:
            return self._lanes.get(oid)
        if lname == LAYER_LANE_CONNECTOR:
            return self._lane_connectors.get(oid)
        if lname == LAYER_ROADBLOCK:
            return self._roadblocks.get(oid)
        if lname == LAYER_ROADBLOCK_CONNECTOR:
            return self._roadblock_connectors.get(oid)
        if lname == LAYER_STOP_LINE:
            return self._stop_lines.get(oid)
        if lname == LAYER_CROSSWALK:
            return self._crosswalks.get(oid)
        if lname == LAYER_CARPARK_AREA:
            return self._carpark_objects.get(oid)
        return None

    def _get_roadblock(self, roadblock_id: Union[str, int]) -> Optional[object]:
        return self._roadblocks.get(str(roadblock_id))

    def _get_roadblock_connector(self, rbc_id: Union[str, int]) -> Optional[object]:
        return self._roadblock_connectors.get(str(rbc_id))

    def get_one_map_object(self, point, layer: LayerLike, search_radius: float = 30.0) -> Optional[object]:
        try:
            px, py = coerce_xy(point)
        except (TypeError, ValueError, AttributeError):
            px, py = point
        p = Point(px, py)
        lname = _to_layer_name(layer)
        tree, geoms, ids, store = self._select_tree_and_store(lname)
        if tree is None:
            return None
        query_geom = p.buffer(search_radius).envelope
        candidates = list(tree.query(query_geom))
        if not candidates:
            query_geom = p.buffer(max(2 * search_radius, 50.0)).envelope
            candidates = list(tree.query(query_geom))
        if not candidates:
            return None
        best_obj = None
        best_d = math.inf
        for item in candidates:
            if isinstance(item, BaseGeometry):
                oid = self._geom_id_lookup(lname, item)
                g = item
            else:
                try:
                    idx = int(item)
                    oid = ids[idx] if idx < len(ids) else None
                    g = geoms[idx] if idx < len(geoms) else None
                except (ValueError, TypeError, IndexError):
                    oid, g = None, None
            obj = store.get(oid) if oid is not None else None
            if obj is None or g is None:
                continue
            d = float(g.distance(p))
            if d < best_d:
                best_d = d
                best_obj = obj
        return best_obj

    def get_distance_to_nearest_map_object(self, point, layer: LayerLike, search_radius: float = 50.0) -> Tuple[Optional[str], float]:
        try:
            px, py = coerce_xy(point)
        except (TypeError, ValueError, AttributeError):
            px, py = point
        p = Point(px, py)
        lname = _to_layer_name(layer)
        tree, geoms, ids, _ = self._select_tree_and_store(lname)
        if tree is None:
            return None, math.inf
        query_geom = p.buffer(search_radius).envelope
        candidates = list(tree.query(query_geom))
        if not candidates:
            query_geom = p.buffer(2 * search_radius).envelope
            candidates = list(tree.query(query_geom))
        if not candidates:
            return None, math.inf
        best_id = None
        best_d = math.inf
        for item in candidates:
            if isinstance(item, BaseGeometry):
                oid = self._geom_id_lookup(lname, item)
                g = item
            else:
                try:
                    idx = int(item)
                    oid = ids[idx] if idx < len(ids) else None
                    g = geoms[idx] if idx < len(geoms) else None
                except (ValueError, TypeError, IndexError):
                    oid, g = None, None
            if oid is None or g is None:
                continue
            d = float(g.distance(p))
            if d < best_d:
                best_d = d
                best_id = oid
        return best_id, best_d

    def _select_tree_and_store(self, lname: str) -> Tuple[Optional[STRtree], List[BaseGeometry], List[str], Dict[str, object]]:
        if lname == LAYER_LANE:
            return self._lane_tree, self._lane_geoms, self._lane_ids, self._lanes
        if lname == LAYER_LANE_CONNECTOR:
            return self._lc_tree, self._lc_geoms, self._lc_ids, self._lane_connectors
        if lname == LAYER_ROADBLOCK:
            return self._rb_tree, self._rb_geoms, self._rb_ids, self._roadblocks
        if lname == LAYER_ROADBLOCK_CONNECTOR:
            return self._rbc_tree, self._rbc_geoms, self._rbc_ids, self._roadblock_connectors
        if lname == LAYER_STOP_LINE:
            return self._sl_tree, self._sl_geoms, self._sl_ids, self._stop_lines
        if lname == LAYER_CROSSWALK:
            return self._cw_tree, self._cw_geoms, self._cw_ids, self._crosswalks
        if lname == LAYER_CARPARK_AREA:
            return self._carpark_tree, self._carpark_geoms, self._carpark_ids, self._carpark_objects
        return None, [], [], {}

    def _geom_id_lookup(self, lname: str, geom: BaseGeometry) -> Optional[str]:
        gid = id(geom)
        if lname == LAYER_LANE:
            return self._lane_gid_map.get(gid)
        if lname == LAYER_LANE_CONNECTOR:
            return self._lc_gid_map.get(gid)
        if lname == LAYER_ROADBLOCK:
            return self._rb_gid_map.get(gid)
        if lname == LAYER_ROADBLOCK_CONNECTOR:
            return self._rbc_gid_map.get(gid)
        if lname == LAYER_STOP_LINE:
            return self._sl_gid_map.get(gid)
        if lname == LAYER_CROSSWALK:
            return self._cw_gid_map.get(gid)
        if lname == LAYER_CARPARK_AREA:
            return self._carpark_gid_map.get(gid)
        return None

    def get_lane_successors(self, lane_id: Union[str, int]) -> List[str]:
        return list(self._lane_successors.get(str(lane_id), []))

    def get_lane_predecessors(self, lane_id: Union[str, int]) -> List[str]:
        return list(self._lane_predecessors.get(str(lane_id), []))

    def get_lane_polygon(self, lane_id: Union[str, int]) -> Optional[BaseGeometry]:
        lane = self._lanes.get(str(lane_id))
        return lane.polygon if lane else None

    def get_lane_baseline(self, lane_id: Union[str, int], sampled: bool = True) -> Optional[BaseGeometry]:
        lane = self._lanes.get(str(lane_id))
        if not lane:
            return None
        if sampled and lane.baseline_sampled is not None:
            return lane.baseline_sampled
        return lane.baseline

    def get_connector_baseline(self, connector_id: Union[str, int], sampled: bool = True) -> Optional[BaseGeometry]:
        lc = self._lane_connectors.get(str(connector_id))
        if not lc:
            return None
        if sampled and lc.baseline_sampled is not None:
            return lc.baseline_sampled
        return lc.baseline

    def get_lane_speed_limit(self, lane_id: Union[str, int]) -> Optional[float]:
        lane = self._lanes.get(str(lane_id))
        return lane.speed_limit_mps if lane else None

    def get_lane_width(self, lane_id: Union[str, int]) -> Optional[float]:
        lane = self._lanes.get(str(lane_id))
        return lane.width_m if lane else None

    def get_roadblock_interior_edges(self, roadblock_id: Union[str, int]) -> List[str]:
        rb = self._roadblocks.get(str(roadblock_id))
        return list(rb.interior_edges) if rb else []

    def get_rbc_interior_edges(self, rbc_id: Union[str, int]) -> List[str]:
        rbc = self._roadblock_connectors.get(str(rbc_id))
        return list(rbc.interior_edges) if rbc else []

    def get_connectors_from_lane(self, lane_id: Union[str, int]) -> List[str]:
        return list(self._from_lane_to_connectors.get(str(lane_id), []))

    def get_connectors_between(self, from_lane_id: Union[str, int], to_lane_id: Union[str, int]) -> List[str]:
        return list(self._pair_to_connectors.get((str(from_lane_id), str(to_lane_id)), []))

    def get_nearest_lane(self, point_xy: Tuple[float, float], search_radius: float = 50.0) -> Tuple[Optional[str], float]:
        """
        Args:
            point_xy: 쿼리 기준점 (x, y) - UTM 좌표계
            search_radius: 검색 반경 (미터)
        
        Returns:
            (차선 ID, 거리) 튜플. 찾지 못한 경우 (None, math.inf)
        """
        lane_id, dist = self.get_distance_to_nearest_map_object(point_xy, LAYER_LANE, search_radius)
        return lane_id, dist

    def project_onto_lane(self, lane_id: Union[str, int], point_xy: Tuple[float, float], sampled: bool = True) -> Optional[Tuple[float, Tuple[float, float], float]]:
        """점을 차선 기준선에 투영. (s_meters_along, foot_point_xy, euclidean_distance) 반환."""
        baseline = self.get_lane_baseline(lane_id, sampled=sampled)
        if baseline is None or baseline.is_empty:
            return None
        p = Point(point_xy[0], point_xy[1])
        try:
            s = float(baseline.project(p))
            foot = baseline.interpolate(s)
            d = float(foot.distance(p))
            return s, (float(foot.x), float(foot.y)), d
        except (ValueError, TypeError, AttributeError):
            return None

    def _line_to_path(self, line: Optional[BaseGeometry]) -> Path:
        if line is None or line.is_empty:
            return Path([])
        coords = list(line.coords)
        if len(coords) == 0:
            return Path([])
        states: List[StateSE2] = []
        for i, (x, y) in enumerate(coords):
            if i == 0:
                nx, ny = coords[1] if len(coords) > 1 else (x, y)
                dx, dy = nx - x, ny - y
            else:
                px, py = coords[i - 1]
                dx, dy = x - px, y - py
            hd = math.atan2(dy, dx) if dx != 0.0 or dy != 0.0 else 0.0
            states.append(StateSE2(float(x), float(y), float(hd)))
        return Path(states)

    def _wire_graphs(self) -> None:
        for lid, lane in self._lanes.items():
            lane.outgoing_edges = [self._lanes[sid] for sid in self._lane_successors.get(lid, []) if sid in self._lanes]
            lane.incoming_edges = [self._lanes[pid] for pid in self._lane_predecessors.get(lid, []) if pid in self._lanes]

        for rb_id, rb in self._roadblocks.items():
            rb.interior_edge_objs = [self._lanes[lid] for lid in rb.interior_edges if lid in self._lanes]

        for rbc_id, rbc in self._roadblock_connectors.items():
            rbc.interior_edge_objs = [self._lane_connectors[cid] for cid in rbc.interior_edges if cid in self._lane_connectors]

    def get_lane_direction_info(self, lane_id: Union[str, int]) -> Dict[str, any]:
        """차선 방향 및 노드 정보 반환."""
        lane = self._lanes.get(str(lane_id))
        if not lane:
            return {}
        return {
            'is_bidirectional': lane.is_bidirectional,
            'start_node_id': lane.start_node_id,
            'end_node_id': lane.end_node_id,
            'junction': lane.junction,
            'lane_type': lane.lane_type,
            'sub_type': lane.sub_type,
            'vehicle_traffic_light_id': lane.vehicle_traffic_light_id,
            'left_link_id': lane.left_link_id,
            'right_link_id': lane.right_link_id
        }
    
    def get_laneside(self, laneside_id: int) -> Optional[Laneside]:
        return self._lanesides.get(laneside_id)
    
    def get_lanesides_by_mid(self, mid: int) -> List[Laneside]:
        return [ls for ls in self._lanesides.values() if ls.mid == mid]
    
    def get_roadlight(self, roadlight_id: int) -> Optional[RoadLight]:
        return self._roadlights.get(roadlight_id)
    
    def get_roadlights_by_lane(self, lane_id: Union[str, int]) -> List[RoadLight]:
        lane_id_int = int(lane_id)
        return [rl for rl in self._roadlights.values() if rl.lane_id == lane_id_int]
    
    def get_pedestrian_light(self, pedestrian_light_id: int) -> Optional[PedestrianLight]:
        return self._pedestrian_lights.get(pedestrian_light_id)
    
    def get_pedestrian_lights_by_crosswalk(self, crosswalk_id: int) -> List[PedestrianLight]:
        return [pl for pl in self._pedestrian_lights.values() if crosswalk_id in pl.crosswalk_ids]
    
    def get_traffic_light_for_lane(self, lane_id: Union[str, int]) -> Optional[RoadLight]:
        lane = self._lanes.get(str(lane_id))
        if not lane or not lane.vehicle_traffic_light_id:
            return None
        return self._roadlights.get(lane.vehicle_traffic_light_id)
    
    def is_lane_junction(self, lane_id: Union[str, int]) -> bool:
        lane = self._lanes.get(str(lane_id))
        return lane is not None and lane.junction is not None and lane.junction > 0
    
    def get_adjacent_lanes(self, lane_id: Union[str, int]) -> Dict[str, Optional[str]]:
        lane = self._lanes.get(str(lane_id))
        if not lane:
            return {'left': None, 'right': None}
        
        left_lane_id = str(lane.left_link_id) if lane.left_link_id and lane.left_link_id > 0 else None
        right_lane_id = str(lane.right_link_id) if lane.right_link_id and lane.right_link_id > 0 else None
        
        return {
            'left': left_lane_id if left_lane_id in self._lanes else None,
            'right': right_lane_id if right_lane_id in self._lanes else None
        }
