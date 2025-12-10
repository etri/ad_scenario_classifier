"""
파일 I/O 유틸리티 모듈

이 모듈은 파일 및 디렉토리 관련 유틸리티 함수들을 제공합니다.
- 디렉토리 생성 (에러 처리 포함)
- 파일 경로 검증
- 절대 경로 변환

참조:
    - doc/01_architecture.md: 파일 I/O 처리 설명
"""

import os
from typing import Optional


def ensure_directory(dir_path: str) -> bool:
    """
    디렉토리를 생성합니다 (이미 존재하면 무시).
    
    디렉토리가 없으면 생성하고, 이미 존재하면 아무 작업도 하지 않습니다.
    에러가 발생하면 False를 반환하고 에러 메시지를 출력합니다.
    
    Args:
        dir_path (str): 생성할 디렉토리 경로
            빈 문자열인 경우 아무 작업도 하지 않고 True를 반환합니다.
    
    Returns:
        bool: 디렉토리 생성 성공 여부
            True: 성공 (디렉토리가 존재하거나 생성됨)
            False: 실패 (권한 오류, 디스크 공간 부족 등)
    
    Raises:
        이 함수는 예외를 발생시키지 않고 False를 반환합니다.
        에러 메시지는 표준 출력에 출력됩니다.
    
    사용 예시:
        success = ensure_directory("./output/scenarios")
        if not success:
            print("디렉토리 생성 실패")
    
    참조:
        - ScenarioExporter: 출력 디렉토리 생성
        - CustomScenarioVisualizer: 이미지 저장 디렉토리 생성
    """
    if not dir_path:  # 빈 문자열인 경우
        return True
    
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except PermissionError as e:
        print(f"   Warning: Permission denied when creating directory: {dir_path}")
        print(f"   Details: {e}")
        return False
    except OSError as e:
        print(f"   Warning: OS error when creating directory: {dir_path}")
        print(f"   Details: {e}")
        return False


def validate_file_path(file_path: str, file_type: str = "file") -> None:
    """
    파일 경로를 검증합니다.
    
    파일이 존재하는지, 파일인지(디렉토리가 아닌지) 등을 검증합니다.
    검증 실패 시 적절한 예외를 발생시킵니다.
    
    Args:
        file_path (str): 검증할 파일 경로
        file_type (str): 파일 타입 설명 (에러 메시지에 사용)
            기본값: "file"
            예: "file", "SQLite database file", "JSON log file"
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: 경로가 파일이 아닌 경우 (디렉토리 등)
    
    사용 예시:
        try:
            validate_file_path("data/map.sqlite", "SQLite database file")
        except FileNotFoundError as e:
            print(f"파일을 찾을 수 없습니다: {e}")
        except ValueError as e:
            print(f"잘못된 경로입니다: {e}")
    
    참조:
        - MapManager: SQLite 데이터베이스 파일 검증
        - JsonLogLoader: JSON 로그 파일 검증 (load 메서드에서 사용)
    """
    if not os.path.exists(file_path):
        abs_path = os.path.abspath(file_path)
        raise FileNotFoundError(
            f"{file_type.capitalize()} not found: {file_path}\n"
            f"   Absolute path: {abs_path}\n"
            f"   Please check if the file exists and the path is correct."
        )
    
    if not os.path.isfile(file_path):
        abs_path = os.path.abspath(file_path)
        raise ValueError(
            f"{file_type.capitalize()} path is not a file: {file_path}\n"
            f"   Absolute path: {abs_path}"
        )


def get_absolute_path(file_path: str) -> str:
    """
    파일 경로의 절대 경로를 반환합니다.
    
    에러 메시지 등에서 사용하기 위한 절대 경로를 반환합니다.
    
    Args:
        file_path (str): 파일 경로 (상대 경로 또는 절대 경로)
    
    Returns:
        str: 절대 경로
    
    사용 예시:
        abs_path = get_absolute_path("./data/log.json")
        print(f"Absolute path: {abs_path}")
        # 출력: Absolute path: /home/user/project/data/log.json
    
    참조:
        - 에러 메시지에서 절대 경로 표시 시 사용
    """
    return os.path.abspath(file_path)

