import logging
import logging.handlers
import os
from datetime import datetime
import pytz

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO):
    """
    로거 설정 함수
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)
        level: 로그 레벨
    
    Returns:
        설정된 로거 객체
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 이미 핸들러가 설정되어 있다면 중복 설정 방지
    if logger.handlers:
        return logger
    
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    
    # 로그 포맷 설정 (한국 시간 포함)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %Z'
    )
    
    # 콘솔 핸들러 (항상 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (log_file이 지정된 경우)
    if log_file:
        # 로그 디렉토리 생성
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 로그 파일 핸들러 (일별 로테이션)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=30,  # 30일간 보관
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_milvus_logger():
    """
    Milvus 서비스용 로거 반환
    """
    # 로그 파일 경로 설정
    log_dir = "logs"
    log_file = os.path.join(log_dir, "milvus_service.log")
    
    return setup_logger("MilvusService", log_file, logging.INFO)


def get_user_activity_logger():
    """
    사용자 행위 추적용 로거 반환
    """
    # 로그 파일 경로 설정
    log_dir = "logs"
    log_file = os.path.join(log_dir, "user_activity.log")
    
    return setup_logger("UserActivity", log_file, logging.INFO)

# 한국 시간대 설정을 위한 유틸리티 함수
def get_korea_time():
    """현재 한국 시간을 반환"""
    korea_tz = pytz.timezone('Asia/Seoul')
    return datetime.now(korea_tz)

def format_korea_time(dt: datetime = None):
    """한국 시간을 포맷팅하여 반환"""
    if dt is None:
        dt = get_korea_time()
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')
