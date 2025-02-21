from typing import Dict, List, Optional
from langchain.agents import Tool, AgentExecutor, AgentType, initialize_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import json
import logging
from dotenv import load_dotenv
import os
import base64
from app.utils.email_sender import send_email
from resend import Emails
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pdfkit

logger = logging.getLogger(__name__)


class ResumeData(BaseModel):
    name: str
    email: str
    phone: str
    education: List[dict]
    experience: List[dict]
    desired_job: str
    skills: str
    additional_info: Optional[str] = ""
    age: Optional[int] = None


class ResumeAgent:
    def __init__(self):
        load_dotenv()

        # LLM 초기화
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            temperature=0.7,
        )

        # 메모리 초기화
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # 자기소개서 생성 프롬프트 초기화
        self.intro_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                시니어 구직자의 자기소개서를 작성해주세요.

                [필수 조건]
                - 총 길이: 600자 내외로 작성
                - 자연스러운 문장 흐름
                - 문단 구분 없이 하나의 완성된 글로 작성
                - 공손하고 예의 바른 어투 사용
                - 제공된 정보만을 바탕으로 작성
                - 입력되지 않은 정보는 언급하지 않음
                - 과장되거나 거짓된 내용 절대 포함하지 않음
                - 현학적인 영어 단어 사용 금지

                [작성 방향]
                1. 실제 입력된 경험과 이력만 포함
                2. 구체적인 업무 경험과 책임감 있게 서술
                3. 입력된 기술과 장점을 바탕으로 작성
                4. 성실하고 진실된 태도 표현
                5. 실제 보유한 능력만 언급

                [유의사항]
                - 입력된 나이 그대로 사용 (과장 금지)
                - 실제 데이터에 기반한 구체적 내용만 작성
                - 추측성 내용이나 과장된 표현 사용하지 않기
                - 입력된 경력과 기술에만 초점 맞추기
                """,
                ),
                ("human", "{resume_data}"),
            ]
        )

        # 도구 정의
        self.tools = [
            Tool(
                name="format_resume",
                description="이력서 데이터를 HTML 형식으로 변환",
                func=self.format_resume,
            ),
            Tool(
                name="generate_introduction",
                description="자기소개서 생성",
                func=self.generate_introduction,
            ),
            Tool(
                name="send_email",
                description="이메일 전송",
                func=self.send_email,
            ),
        ]

        # Agent 초기화
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
        )

    async def filter_resume_data(self, resume_data: dict) -> dict:
        """이력서 데이터 필터링 및 가공"""
        try:
            filtered_data = {}

            # 개인정보는 그대로 유지
            personal_info = {}
            for key in ["name", "email", "phone"]:
                if resume_data.get(key):
                    personal_info[key] = str(resume_data[key])
            if personal_info:
                filtered_data["personal_info"] = personal_info

            # 보유 기술 및 장점 가공
            if resume_data.get("skills"):
                skills_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
                        입력된 기술이나 장점을 자연스러운 문장으로 가공해주세요.
                        
                        [규칙]
                        - 각 항목을 구체적으로 설명
                        - 실무적인 관점에서 장점 부각
                        - 간단명료한 문장 사용
                        - 과장된 표현 사용하지 않기
                        
                        예시 입력: "강한 체력, 성실함"
                        예시 출력: "체력적으로 건강하여 장시간 서서 일하는 업무도 무리없이 수행할 수 있으며, 성실한 태도로 맡은 업무를 책임감 있게 완수합니다."
                        """,
                        ),
                        ("human", resume_data["skills"]),
                    ]
                )

                skills_response = await self.llm.ainvoke(
                    skills_prompt.format_messages(skills=resume_data["skills"])
                )
                filtered_data["skills"] = skills_response.content

            # 경력사항 가공
            if "experience" in resume_data and isinstance(
                resume_data["experience"], list
            ):
                experience = []
                for exp in resume_data["experience"]:
                    if isinstance(exp, dict) and any(exp.values()):
                        if exp.get("description"):
                            desc_prompt = ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        """
                                    입력된 업무 설명을 전문적이고 구체적인 문장으로 가공해주세요.
                                    
                                    [규칙]
                                    - 성과와 책임을 구체적으로 기술
                                    - 업무 프로세스 중심으로 설명
                                    - 간결하고 명확한 문장 사용
                                    - 실제 수행한 업무 위주로 작성
                                    
                                    예시 입력: "마트 진열 담당"
                                    예시 출력: "상품 진열 및 재고 관리를 담당하여 고객이 쉽게 상품을 찾을 수 있도록 체계적으로 진열 작업을 수행했습니다. 신선도 유지가 필요한 상품은 특별히 관리하여 품질 저하를 방지했습니다."
                                    """,
                                    ),
                                    ("human", exp["description"]),
                                ]
                            )

                            desc_response = await self.llm.ainvoke(
                                desc_prompt.format_messages(
                                    description=exp["description"]
                                )
                            )
                            exp["description"] = desc_response.content
                        experience.append(exp)
                if experience:
                    filtered_data["experience"] = experience

            # 학력사항은 그대로 유지
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

            # 희망 직종은 그대로 유지
            if resume_data.get("desired_job"):
                filtered_data["desired_job"] = resume_data["desired_job"]

            return filtered_data

        except Exception as e:
            logger.error(f"데이터 필터링 중 오류: {str(e)}")
            raise

    async def format_resume(self, data: dict) -> str:
        """이력서 HTML 생성"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
        주어진 이력서 데이터를 전문적으로 가공하여 HTML로 생성해주세요.

        [작성 규칙]
            1. 데이터를 전문적으로 가공하여 작성
            2. 보유 기술과 장점은 구체적이고 전문적인 표현으로 변환
            3. 자기소개서는 전문적이고 구체적인 표현으로 보완
            4. 각 섹션의 내용을 보완하고 확장
            5. 전문적이고 공손한 어투 사용
            6. 성과와 역량 중심의 표현
            
            [HTML 형식]
            <div class="resume">
                <h1>이력서</h1>
            <!-- 각 섹션 -->
                <section>
                    <h2>[섹션 제목]</h2>
                [전문적으로 가공된 내용]
            </section>
        </div>
            """,
                ),
                ("human", "{data}"),
            ]
        )

        response = await self.llm.ainvoke(
            prompt.format_messages(data=json.dumps(data, ensure_ascii=False))
        )
        return response.content

    async def generate_introduction(self, resume_data: dict) -> dict:
        """자기소개서 생성"""
        try:
            logger.info("자기소개서 생성 시작")
            logger.info(f"입력 데이터: {resume_data}")

            # 데이터 필터링 및 가공
            filtered_data = await self.filter_resume_data(resume_data)
            logger.info(f"필터링된 데이터: {filtered_data}")

            try:
                # 자기소개서 생성
                messages = self.intro_prompt.format_messages(
                    resume_data=json.dumps(filtered_data, ensure_ascii=False)
                )
                intro_response = await self.llm.ainvoke(messages)

                # 응답 내용 확인 및 변환
                if not intro_response or not hasattr(intro_response, "content"):
                    raise ValueError("AI 응답이 올바르지 않습니다")

                # 응답 데이터 구조화
                response_data = {
                    "status": "success",
                    "data": {
                        "content": intro_response.content,
                        "filtered_data": {
                            "experience": filtered_data.get("experience", []),
                            "skills": filtered_data.get("skills", ""),
                        },
                    },
                }

                logger.info(f"최종 응답 데이터: {response_data}")
                return response_data

            except Exception as api_error:
                logger.error(f"OpenAI API 호출 중 오류: {str(api_error)}")
                return {"status": "error", "message": str(api_error)}

        except Exception as e:
            logger.error(f"자기소개서 생성 중 오류: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def generate_email_content(self, resume_data: dict) -> tuple[str, str]:
        """이력서 데이터를 기반으로 이메일 제목과 내용 생성"""
        prompt = f"""
        다음 이력서 정보를 바탕으로 지원 이메일의 제목과 내용을 작성해주세요.
        
        지원자 정보:
        - 이름: {resume_data.get('name')}
        - 희망 직무: {resume_data.get('desired_job')}
        - 학력: {resume_data.get('education', [])[0].get('school') if resume_data.get('education') else ''}
        - 경력: {resume_data.get('experience', [])[0].get('company') if resume_data.get('experience') else ''}
        - 보유 기술: {resume_data.get('skills')}
        
        다음 형식으로 작성해주세요:
        제목: [이력서 지원] {resume_data.get('desired_job')} 포지션 지원
        """

        response = await self.llm.ainvoke(prompt)
        content = response.content.split("\n")

        subject = content[0].replace("제목: ", "")
        body = "\n".join(content[2:])

        return subject, body

    async def send_email(self, email: str, name: str, pdf_content: str):
        """이력서 이메일 전송"""
        try:
            response = Emails.send(
                {
                    "from": "SeniorJobGo <onboarding@resend.dev>",
                    "to": email,
                    "subject": f"{name}의 이력서 제출드립니다.",
                    "html": f"{name}의 이력서 제출드립니다.<br><br>감사합니다.",
                    "attachments": [
                        {
                            "filename": f"{name}_이력서.pdf",
                            "content": pdf_content,  # base64로 인코딩된 PDF 데이터
                            "type": "application/pdf",
                        }
                    ],
                }
            )

            return response
        except Exception as e:
            logger.error(f"이메일 전송 중 오류: {str(e)}")
            raise

    async def generate_resume(self, resume_data: dict) -> str:
        """이력서 생성 프로세스 실행"""
        try:
            # 1. 데이터 필터링 및 정리
            filtered_data = await self.filter_resume_data(resume_data)
            if not filtered_data:
                raise ValueError("사용 가능한 이력서 데이터가 없습니다.")

            # 2. 자기소개서 생성
            introduction = await self.generate_introduction(filtered_data)
            filtered_data["introduction"] = introduction

            # 3. HTML 형식의 이력서 생성
            html_content = await self.format_resume(filtered_data)

            return html_content

        except Exception as e:
            logger.error(f"이력서 생성 중 오류 발생: {str(e)}")
            raise

    async def generate_pdf(self, html_content: str) -> BytesIO:
        """HTML을 PDF로 변환"""
        try:
            # 기존 PDF 설정 유지
            config = pdfkit.configuration(
                wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
            )
            options = {
                "encoding": "UTF-8",
                "page-size": "A4",
                "margin-top": "0.75in",
                "margin-right": "0.75in",
                "margin-bottom": "0.75in",
                "margin-left": "0.75in",
                "enable-local-file-access": True,
            }

            # PDF 생성
            pdf_data = pdfkit.from_string(
                html_content,
                False,  # 파일로 저장하지 않고 바이트로 반환
                configuration=config,
                options=options,
            )

            # BytesIO 객체로 변환
            pdf_buffer = BytesIO(pdf_data)
            return pdf_buffer

        except Exception as e:
            logger.error(f"PDF 생성 중 오류: {str(e)}")
            raise

    async def format_resume_content(self, resume_data: dict) -> str:
        """이력서 내용을 HTML 형식으로 구성"""
        try:
            sections = []

            # 기본 정보 섹션
            basic_info = {
                k: v
                for k, v in {
                    "name": resume_data.get("name"),
                    "age": resume_data.get("age"),
                    "phone": resume_data.get("phone"),
                    "email": resume_data.get("email"),
                }.items()
                if v
            }

            # LLM 프롬프트 정의
            base_prompt = """
            입력된 정보를 자연스러운 문장으로 변환해주세요.

            [규칙]
            1. 실제 입력된 정보만 사용
            2. 쉽고 명확한 한글 단어만 사용
            3. 외래어나 영어 사용하지 않기
            4. 입력되지 않은 정보는 언급하지 않음
            5. 순수하게 내용만 자연스러운 문장으로 작성
            """

            # 기본 정보 섹션 HTML 생성
            if basic_info:
                basic_info_html = "<section class='basic-info'><h2>기본 정보</h2>"
                for key, value in basic_info.items():
                    label = {
                        "name": "이름",
                        "age": "나이",
                        "phone": "연락처",
                        "email": "이메일",
                    }[key]
                    unit = "세" if key == "age" else ""
                    basic_info_html += f"<p data-label='{label}'>{value}{unit}</p>"
                basic_info_html += "</section>"
                sections.append(basic_info_html)

            # 자기소개서 섹션
            if resume_data.get("additional_info"):
                intro_prompt = f"{base_prompt}\n\n다음 내용을 자연스럽게 서술해주세요:\n{resume_data['additional_info']}"
                intro_html = await self.llm.apredict(intro_prompt)
                intro_html = (
                    intro_html.replace("자기소개서:", "")
                    .replace("**자기소개서**", "")
                    .replace("*", "")
                    .strip()
                )

                if intro_html.strip():
                    sections.append(
                        f"""
                        <section class='additional-info'>
                            <h2>자기소개서</h2>
                            <div class='content'>{intro_html}</div>
                        </section>
                        """
                    )

            # 경력 사항 섹션
            if resume_data.get("experience") and any(
                exp for exp in resume_data["experience"] if any(exp.values())
            ):
                exp_prompt = f"{base_prompt}\n\n다음 경력을 자연스럽게 서술해주세요:\n{json.dumps(resume_data['experience'], ensure_ascii=False)}"
                experience_html = await self.llm.apredict(exp_prompt)
                experience_html = (
                    experience_html.replace("경력 사항:", "")
                    .replace("**경력 사항**", "")
                    .replace("*", "")
                    .strip()
                )

                if experience_html.strip():
                    sections.append(
                        f"""
                        <section class='experience'>
                            <h2>경력 사항</h2>
                            <div class='content'>{experience_html}</div>
                        </section>
                        """
                    )
            else:
                sections.append(
                    """
                    <section class='experience'>
                        <h2>경력 사항</h2>
                        <div class='content'>신입으로 지원합니다.</div>
                    </section>
                    """
                )

            # 보유 기술 섹션
            if resume_data.get("skills"):
                skills_prompt = """
                입력된 기술과 장점을 자연스럽게 서술해주세요.
                
                [규칙]
                1. 간단명료하게 작성
                2. 실제 보유한 능력만 언급
                3. 과장된 표현 사용하지 않기
                4. 구체적인 예시 포함
                """

                skills_html = await self.llm.apredict(
                    f"{skills_prompt}\n\n{resume_data['skills']}"
                )
                skills_html = (
                    skills_html.replace("보유 기술:", "")
                    .replace("보유 기술 및 장점:", "")
                    .replace("**보유 기술**", "")
                    .replace("*", "")
                    .strip()
                )

                if skills_html.strip():
                    sections.append(
                        f"""
                        <section class='skills'>
                            <h2>보유 기술 및 장점</h2>
                            <div class='content'>{skills_html}</div>
                        </section>
                        """
                    )

            # 최종 HTML 구성
            html_content = f"""
            <div class="resume">
                <h1>이력서</h1>
                {''.join(sections)}
            </div>
            """

            return html_content

        except Exception as e:
            logger.error(f"이력서 HTML 생성 중 오류: {str(e)}")
            raise

    async def _get_formatted_content(self, resume_data: dict) -> dict:
        """이력서 내용을 자연스럽게 구성"""
        try:
            # 나이에 따른 프롬프트 조정
            age = resume_data.get("age")
            age_context = f"""
            - {age}대의 풍부한 경험과 지혜를 강조
            - 성실성과 책임감 있는 태도 부각
            - 안정적이고 신중한 업무 처리 능력 강조
            - 풍부한 대인관계 능력과 소통 강조
            - 시간 여유가 많아 안정적인 근무 가능
            - 오랜 경험에서 나오는 노하우 강조
            """

            prompt = f"""
            {age}대 시니어 구직자의 이력서를 작성합니다.
            주어진 이력서 데이터를 바탕으로 자연스러운 문장으로 구성해주세요.

            [지원자 나이: {age}세]

            [규칙]
            1. 경력 사항은 업무 성과와 책임을 구체적으로 서술
            2. 보유 기술은 실무 경험과 연계하여 설명
            3. 각 항목은 완성된 문장으로 작성
            4. 불필요한 과장이나 추측성 내용 제외
            5. 시니어 구직자의 강점 부각
            6. 입력된 데이터만 사용

            [시니어 구직자의 강점]
            {age_context}

            [필수 포함 요소]
            - 풍부한 경험과 지혜
            - 책임감 있는 업무 수행 능력
            - 원활한 소통과 협력
            - 안정적인 근무 가능성
            """

            formatted_data = {
                "basic_info": {
                    "name": resume_data.get("name", ""),
                    "age": resume_data.get("age"),
                    "phone": resume_data.get("phone", ""),
                    "email": resume_data.get("email", ""),
                }
            }

            # 경력 사항 포맷팅
            if resume_data.get("experience"):
                experience_prompt = f"{prompt}\n\n경력 사항:\n{json.dumps(resume_data['experience'], ensure_ascii=False)}"
                experience_response = await self.llm.apredict(experience_prompt)
                formatted_data["experience"] = experience_response

            # 보유 기술 포맷팅
            if resume_data.get("skills"):
                skills_prompt = f"{prompt}\n\n보유 기술:\n{resume_data['skills']}"
                skills_response = await self.llm.apredict(skills_prompt)
                formatted_data["skills"] = skills_response

            # 자기소개서 포맷팅
            if resume_data.get("additional_info"):
                intro_prompt = (
                    f"{prompt}\n\n자기소개서:\n{resume_data['additional_info']}"
                )
                intro_response = await self.llm.apredict(intro_prompt)
                formatted_data["additional_info"] = intro_response

            return formatted_data

        except Exception as e:
            logger.error(f"이력서 내용 포맷팅 중 오류: {str(e)}")
            raise
