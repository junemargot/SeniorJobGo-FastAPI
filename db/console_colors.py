"""
콘솔 출력 색상 관련 코드입니다.
"""

# ANSI 색상 코드
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# 색상 입혀서 콘솔 출력
INFO = f"{GREEN}INFO{RESET}:     "
ERROR = f"{RED}ERROR{RESET}:    " 