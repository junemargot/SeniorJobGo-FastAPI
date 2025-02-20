from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import json
import traceback


class ResumeAgent:
    def __init__(self):
        load_dotenv()
        try:
            self.llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-4",
                temperature=0.7,
            )
        except Exception as e:
            print(f"LLM 초기화 실패: {str(e)}")
            raise

        self.tools = [
            Tool(
                name="get_resume_data",
                description="입력된 이력서 데이터를 가져오는 도구",
                func=self.get_resume_data,
            ),
            Tool(
                name="format_resume",
                description="이력서 데이터를 HTML 형식으로 변환하는 도구",
                func=self.format_resume,
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def get_resume_data(self, resume_data: dict) -> dict:
        """입력된 이력서 데이터를 필터링하고 정리하여 반환"""
        try:
            print("원본 데이터:", resume_data)  # 디버깅용

            if not isinstance(resume_data, dict):
                raise ValueError(f"잘못된 데이터 형식입니다: {type(resume_data)}")

            filtered_data = {}

            # 개인정보 필터링
            personal_info = {}
            for key in ["name", "email", "phone"]:
                if resume_data.get(key):
                    personal_info[key] = str(resume_data[key])
            if personal_info:
                filtered_data["personal_info"] = personal_info

            # 학력사항 필터링
            if "education" in resume_data and isinstance(
                resume_data["education"], list
            ):
                education = [
                    edu
                    for edu in resume_data["education"]
                    if isinstance(edu, dict) and any(edu.values())
                ]
                if education:
                    filtered_data["education"] = education

            # 경력사항 필터링
            if "experience" in resume_data and isinstance(
                resume_data["experience"], list
            ):
                experience = [
                    exp
                    for exp in resume_data["experience"]
                    if isinstance(exp, dict) and any(exp.values())
                ]
                if experience:
                    filtered_data["experience"] = experience

            # 희망 직종 필터링
            if resume_data.get("desired_job"):
                filtered_data["desired_job"] = resume_data["desired_job"]

            # 보유 기술 및 장점 필터링
            if resume_data.get("skills"):
                filtered_data["skills"] = resume_data["skills"]

            # 자기소개서 필터링
            if resume_data.get("additional_info"):
                filtered_data["cover_letter"] = resume_data["additional_info"]

            print("필터링된 데이터:", filtered_data)  # 디버깅용
            return filtered_data

        except Exception as e:
            print(f"데이터 필터링 중 오류: {str(e)}")
            raise

    def format_resume(self, data: dict) -> str:
        """LLM을 사용하여 이력서 HTML 생성"""
        prompt = f"""
        주어진 이력서 데이터를 전문적으로 가공하여 HTML로 생성해주세요.
        
        데이터:
        {json.dumps(data, indent=2, ensure_ascii=False)}

        [작성 규칙]
        1. 데이터를 있는 그대로 쓰지 말고 전문적으로 가공해서 작성해주세요.
        2. 보유 기술과 장점은 구체적이고 전문적인 표현으로 바꿔주세요.
            예시) "체력 강함" -> "장시간 서서 일하는 업무에 적합한 체력 보유"
        3. 자기소개서는 더 전문적이고 구체적인 표현으로 다듬어주세요.
        4. 각 섹션의 내용을 보완하고 확장해서 작성해주세요.
        5. 전문적이고 공손한 어투를 사용해주세요.
        6. 다음 표현은 사용하지 마세요:
            - "~드립니다", "~올립니다"와 같은 맺음말
            - "~을/를 드리다"와 같은 높임말
            - "감사합니다", "잘 부탁드립니다"와 같은 관용구
        7. 대신 다음과 같이 작성해주세요:
            - 사실 중심의 객관적인 서술
            - "~했습니다", "~하고 있습니다"와 같은 중립적인 종결어
            - 성과와 역량 중심의 표현

        [HTML 형식]
        <div class="resume" style="font-family: Arial, sans-serif; color: #333333; line-height: 1.6;">
            <h1 style="font-size: 24px; margin-bottom: 30px; text-align: center;">이력서</h1>
            
            <!-- 각 섹션 -->
            <section style="margin-bottom: 30px;">
                <h2 style="font-size: 18px; border-bottom: 2px solid #dddddd; padding-bottom: 10px;">[섹션 제목]</h2>
                [전문적으로 가공된 내용]
            </section>
        </div>

        [주의사항]
        1. 모든 내용을 전문적이고 구체적으로 표현해주세요
        2. 설명글이나 주석은 제외하고 순수 이력서 내용만 생성해주세요
        3. 모든 스타일은 인라인 CSS로 작성해주세요
        4. 없는 섹션은 생략해주세요
        """

        response = self.llm.invoke(prompt)
        generated_html = response.content.strip()

        # 코드 블록 마커 제거
        if generated_html.startswith("```html"):
            generated_html = generated_html[7:]
        if generated_html.endswith("```"):
            generated_html = generated_html[:-3]

        return generated_html.strip()

    async def generate_resume(self, resume_data: dict) -> str:
        """이력서 생성 프로세스 실행"""
        try:
            print("받은 데이터:", resume_data)  # 디버깅용

            # 1. 데이터 필터링 및 정리
            filtered_data = self.get_resume_data(resume_data)
            print("필터링된 데이터:", filtered_data)  # 디버깅용

            if not filtered_data:
                raise ValueError("사용 가능한 이력서 데이터가 없습니다.")

            # 2. 필터링된 데이터로 HTML 생성
            html_content = self.format_resume(filtered_data)
            print("생성된 HTML 길이:", len(html_content))  # 디버깅용

            return html_content

        except Exception as e:
            print(f"이력서 생성 중 오류 발생: {str(e)}")
            raise
