"""
본 프로젝트에서 사용할 모델들을 정의합니다.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ChatModel(BaseModel):
    index: int = Field(..., ge=0)  # 대화 인덱스. 0 이상
    role: str = Field(..., pattern="^(user|assistant)$")  # 대화 역할. 'user' 또는 'assistant'만 허용
    content: str = Field(..., min_length=1)  # 대화 내용. 최소 1자
    created_at: datetime = Field(default_factory=datetime.now)  # 대화 생성 시간

class UserModel(BaseModel):
    id: str = Field(..., min_length=3, max_length=50)  # 사용자 아이디. 최소 3자, 최대 50자
    password: Optional[str] = Field(None, min_length=8)  # 사용자 비밀번호. 최소 8자
    provider: str = Field(..., pattern="^(none|local|kakao)$")  # 로그인 방법. 'none'은 비로그인, 'local'은 기본 로그인, 'kakao'는 카카오 로그인
    name: str = Field(..., min_length=1, max_length=100)  # 이름. 최소 1자, 최대 100자
    birth_year: Optional[int] = None  # 출생년도
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")  # 성별. 'male', 'female', 'other'만 허용
    hope_job: Optional[str] = Field(None, max_length=100)  # 희망 직종. 최대 100자
    hope_location: Optional[str] = Field(None, max_length=100)  # 희망 근무지. 최대 100자
    hope_salary: Optional[int] = None  # 희망 급여. 0 이상
    education: str = Field(..., pattern="^(high_school|college|university|graduate)$")  # 학력. 'high_school', 'college', 'university', 'graduate'만 허용
    messages: List[ChatModel] = []  # 사용자의 채팅 메시지 목록
    created_at: datetime = Field(default_factory=datetime.now)  # 생성 시간

# - 참고
#   - erd: https://www.erdcloud.com/d/Fg2N9YaBH68pPSnsN
#       - 위 사이트를 참고해서 모델을 제작 중입니다. 일부 변경된 부분이 있어 추후 수정이 필요하시다면 말씀해주세요.
#   - UserModel의 messages 필드가 너무 많아서 문제가 될 수 있음
#   - 만약 너무 많다면 메세지 목록을 따로 모델로 만들어서 사용하는 것이 좋음
#   - 예시. 만약 사용하신다고 한다면 Ctrl + Alt + 방향키와 End 키로 코드를 깔끔하게 복사해서 사용할 수 있음
#     ```
#       class MessageModel(BaseModel):
#           id: str  # 사용자 아이디
#           messages: List[ChatModel]  # 채팅 메시지 목록
#     ```
#   - 이후 UserModel의 messages 필드를 MessageModel로 대체하면 됨

# -  Optional 자료형에 대한 설명
#   - 선택적 필드는 필수 필드가 아니라 선택적으로 사용할 수 있는 필드입니다.
#       - 예시. birth_year: Optional[int] = None  # 출생년도
