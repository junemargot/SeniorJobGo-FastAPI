from fastapi import APIRouter, Request, Response, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting, TrainingCourse

from db.database import db
from db.routes_auth import get_user_info_by_cookie
from db.routes_chat import add_chat_message
from bson import ObjectId
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent

import os
import json
import time
from app.routes.training_router import search_training_courses, get_common_code_collector

logger = logging.getLogger(__name__)

router = APIRouter()

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

# 의존성 함수들
def get_llm(request: Request):
    return request.app.state.llm

def get_vector_search(request: Request):
    return request.app.state.vector_search

def get_job_advisor_agent(request: Request):
    return request.app.state.job_advisor_agent

def get_chat_agent(request: Request, llm=Depends(get_llm)):
    return ChatAgent(llm=llm)

@router.post("/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent)
) -> ChatResponse:
    start_time = time.time()
    try:
        # 요청 시작 로깅
        logger.info("="*50)
        logger.info("[ChatRouter] 새로운 채팅 요청 시작")
        logger.info(f"[ChatRouter] 요청 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 사용자 프로필: {chat_request.user_profile}")
        
        # DB 체크
        if db is None:
            logger.error("[ChatRouter] DB 연결 없음")
            raise Exception("DB connection is None")

        # 쿠키에서 사용자 정보 조회
        user = None
        try:
            user = await get_user_info_by_cookie(request)
        except:
            logger.error(f"[ChatRouter] 쿠키에서 사용자 정보 조회 중 오류 발생 - 기본 응답 반환")
            return ChatResponse(
                message=chat_request.user_message,
                jobPostings=[],
                trainingCourses=[],
                type="info",
                user_profile=chat_request.user_profile or {}
            )

        # 이전 대화 이력 가져오기
        chat_history = user.get("messages", [])
        
        # 채용/훈련 정보 관련 대화만 필터링
        formatted_history = ""
        if chat_history:  # chat_history가 있을 때만 필터링 수행
            for i in range(len(chat_history)-1, -1, -1):  # 최신 메시지부터 역순으로 확인
                msg = chat_history[i]
                if isinstance(msg, dict):  # dict 타입 체크 추가
                    role = "사용자" if msg.get("role") == "user" else "시스템"
                    content = msg.get("content", "")
                    # dict인 경우 필요한 정보만 추출
                    if isinstance(content, dict):
                        content = content.get("message", "")
                    elif not isinstance(content, str):
                        content = str(content)
                    formatted_history = f"{role}: {content}\n" + formatted_history

        # 이전 검색 결과가 있는지 확인 (빈 대화 이력 처리)
        prev_message = ""
        if formatted_history:
            history_lines = formatted_history.strip().split("\n")
            for line in reversed(history_lines):
                if line.startswith("시스템:"):
                    prev_message = line.replace("시스템:", "").strip()
                    break

        # 이전 검색에서 훈련과정 관련 키워드가 있는지 확인
        if prev_message and "관련 훈련과정을" in prev_message and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            # 이전 검색 키워드 추출
            keyword = prev_message.split("'")[1] if "'" in prev_message else ""
            if keyword:
                chat_request.user_message = f"{keyword} {chat_request.user_message}"
        
        logger.info("[ChatRouter] JobAdvisorAgent.chat 호출 시작")
        try:
            response = await job_advisor_agent.chat(
                query=chat_request.user_message,
                user_profile=chat_request.user_profile,
                chat_history=formatted_history
            )
            logger.info("[ChatRouter] JobAdvisorAgent.chat 응답 성공")
            logger.info(f"[ChatRouter] 응답 내용: {response}")
            
        except Exception as chat_error:
            logger.error("[ChatRouter] JobAdvisorAgent.chat 실행 중 에러", exc_info=True)
            raise chat_error
        
        # 가격 필터링이 필요한 경우
        if response.get("type") == "training" and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            courses = response.get("trainingCourses", [])
            filtered_courses = []
            for course in courses:
                try:
                    cost = int(course.get("cost", "0").replace(",", "").replace("원", ""))
                    if cost <= 300000:  # 30만원 이하
                        filtered_courses.append(course)
                except ValueError:
                    continue
            
            if filtered_courses:
                response["trainingCourses"] = filtered_courses
                response["message"] = f"30만원 이하의 저렴한 훈련과정 {len(filtered_courses)}개를 찾았습니다."
            else:
                response["message"] = "죄송합니다. 조건에 맞는 저렴한 훈련과정을 찾지 못했습니다."

        # 메시지 저장
        try:
            await add_chat_message(user, chat_request.user_message, response.get("message", ""))
            logger.info("[ChatRouter] 메시지 저장 완료")
        except:
            logger.error("[ChatRouter] 메시지 저장 중 에러", exc_info=True)
            # 메시지 저장 실패는 치명적이지 않으므로 계속 진행

        # 응답 생성
        processing_time = time.time() - start_time
        response["processingTime"] = round(processing_time, 2)

        # JobPosting 목록 생성
        job_postings_list = []
        for idx, jp in enumerate(response.get("jobPostings", [])):
            job_postings_list.append(JobPosting(
                id=jp.get("id", "no_id"),
                location=jp.get("location", ""),
                company=jp.get("company", ""),
                title=jp.get("title", ""),
                salary=jp.get("salary", ""),
                workingHours=jp.get("workingHours", "정보없음"),
                description=jp.get("description", ""),
                phoneNumber=jp.get("phoneNumber", ""),
                deadline=jp.get("deadline", ""),
                requiredDocs=jp.get("requiredDocs", ""),
                hiringProcess=jp.get("hiringProcess", ""),
                insurance=jp.get("insurance", ""),
                jobCategory=jp.get("jobCategory", ""),
                jobKeywords=jp.get("jobKeywords", ""),
                posting_url=jp.get("posting_url", ""),
                rank=jp.get("rank", idx+1)
            ))

        # TrainingCourse 목록 생성
        training_courses_list = []
        for course in response.get("trainingCourses", []):
            training_courses_list.append(TrainingCourse(
                id=course.get("id", ""),
                title=course.get("title", ""),
                institute=course.get("institute", ""),
                location=course.get("location", ""),
                period=course.get("period", ""),
                startDate=course.get("startDate", ""),
                endDate=course.get("endDate", ""),
                cost=course.get("cost", ""),
                description=course.get("description", ""),
                target=course.get("target"),
                yardMan=course.get("yardMan"),
                titleLink=course.get("titleLink"),
                telNo=course.get("telNo")
            ))

        # 최종 응답 생성
        response_model = ChatResponse(
            message=response.get("message", ""),
            jobPostings=job_postings_list,
            trainingCourses=training_courses_list,
            type=response.get("type", "info"),
            user_profile=response.get("user_profile", {})
        )
        
        logger.info("[ChatRouter] 응답 생성 완료")
        logger.info("="*50)
        
        return response_model

    except Exception as e:
        logger.error("[ChatRouter] 치명적인 에러 발생", exc_info=True)
        error_response = ChatResponse(
            type="error",
            message="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            jobPostings=[],
            trainingCourses=[],
            user_profile={}
        )
        return error_response