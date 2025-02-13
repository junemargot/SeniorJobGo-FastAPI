from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from app.utils.constants import DICTIONARY  
import re

# 사전 변환 규칙을 적용하는 함수 (DICTIONARY 직접 사용)
def apply_dictionary_rules(query: str) -> str:
    """사용자의 질문을 사전(DICTIONARY)에 따라 변환하는 함수"""
    pattern = re.compile("|".join(map(re.escape, DICTIONARY.keys())))
    return pattern.sub(lambda match: DICTIONARY[match.group(0)], query)

# 문서 검증 프롬프트
# verify_prompt = PromptTemplate.from_template("""
# 다음 문서들이 사용자의 질문에 답변하기에 충분한 정보를 포함하고 있는지 판단해주세요.

# 질문: {query}

# 문서들:
# {context}

# 답변 형식:
# - 문서가 충분한 정보를 포함하고 있다면 "YES"
# - 문서가 충분한 정보를 포함하고 있지 않다면 "NO"

# 답변:
# """)
verify_prompt = PromptTemplate.from_template("""
Please determine whether the following documents contain enough information to answer the user's question.

Question: {query}

Documents:
{context}

Answer format:
- If the documents contain sufficient information, reply "YES"
- If the documents do not contain sufficient information, reply "NO"

Answer:
""")

# 질문 변환 프롬프트 (DICTIONARY 적용됨)
# rewrite_prompt = PromptTemplate.from_template("""
# 사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
# 이때 반드시 사전에 있는 규칙을 적용해야 합니다.

# 원본 질문: {original_query}

# 변경된 질문: {transformed_query}
# """)
rewrite_prompt = PromptTemplate.from_template("""
Look at the user's question and refer to our dictionary to modify the user's question.
Make sure to strictly apply the rules in the dictionary.

Original question: {original_query}

Modified question: {transformed_query}""")


# 채용 공고 추천 프롬프트
generate_prompt = PromptTemplate.from_template("""
Based on the following information, please compose a helpful response for the job seeker.
Pay special attention to whether each job posting's region matches the region the user is looking for.

Question: {question}

Reference documents:
{context}

Answer format:
Display the discovered job postings in the following card format:

[Separator]
📍 [Region] • [Company Name]
[Job Posting Title]

💰 [Salary Conditions]
⏰ [Working Hours]
📝 [Key Job Duties - summarized in one line]

[Separator]

Show each posting in the above format. Make sure the response is clear and detailed so the job seeker can easily understand it.
""")

# 챗봇 페르소나 설정
chat_persona_prompt = """You are an AI job counselor specializing in assisting senior job seekers.

Persona:
- A friendly counselor with strong empathy.
- Fully understands the characteristics and needs of senior job seekers.
- Uses emojis effectively to create a friendly atmosphere.
- Naturally guides the conversation toward job search information.

Conversation principles:
1. Respond naturally even to casual, everyday conversation, but connect it to job search themes.
2. Include questions to identify the job seeker's preferences and conditions.
3. Use language that is friendly to seniors.
4. Provide clear and easily understandable explanations.
"""

# 기본 대화 프롬프트
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_persona_prompt),
    ("human", "{query}")  # input -> query로 변경하여 일관성 유지
])

# 정보 추출 프롬프트
EXTRACT_INFO_PROMPT = PromptTemplate.from_template("""
You are an expert at extracting job-related information from the user's natural conversation.

Previous conversation:
{chat_history}

Current message: {user_query}

Please extract the job type, region, and age group from the above conversation.
Refer to the previous conversation context to supplement any missing information.

Be aware of various expressions like:
- Job type: "일자리" (jobs), "자리" (position), "일거리" (work), "직장" (workplace), "취직" (getting hired), "취업" (employment)
- Region: "여기" (here), "이 근처" (nearby), "우리 동네" (our neighborhood), "근처" (near), "가까운" (close)
- Age group: "시니어" (senior), "노인" (elderly), "어르신" (senior), "중장년" (middle-aged)

Examples:
1. "서울에서 경비 일자리 좀 알아보려고요" -> {{"직무": "경비", "지역": "서울", "연령대": ""}}
2. "우리 동네 근처에서 할만한 일자리 있나요?" -> {{"직무": "", "지역": "근처", "연령대": ""}}
3. "시니어가 할 만한 요양보호사 자리 있을까요?" -> {{"직무": "요양보호사", "지역": "", "연령대": "시니어"}}

Respond **only** in the following JSON format:
{{
    "직무": "extracted job type (empty string if none)",
    "지역": "extracted region (empty string if none)",
    "연령대": "extracted age group (empty string if none)"
}}

Special rules:
1. Even if the job type is not specific, if terms like "일자리", "일거리", or "자리" are mentioned, treat the job type as an empty string.
2. Standardize all references to "여기", "이 근처", "근처" etc. as "근처" (near).
3. Standardize all senior-related expressions (시니어, 노인, 어르신, 중장년) as "시니어".
4. Use previous conversation information to understand the current context.

""")

# 의도 분류 프롬프트 수정
CLASSIFY_INTENT_PROMPT = PromptTemplate.from_template("""
You are an expert career counselor specializing in senior job seekers, capable of accurately identifying the user's intent, especially hidden intentions related to job search or vocational training.

Previous conversation:
{chat_history}

Current message: {user_query}

Intents to classify:
1. job (related to job seeking)
   - Contains words like 일자리/직장/취업/채용/자리
   - Mentions of a specific region or position (e.g., "Seoul", "경비" for security guard, "요양보호사" for caregiver)
   - Mentions of age/experience/job requirements
   - Inquiries about salary or working hours
   - Any expression of wanting a job

2. training (related to vocational training)
   - Words like 교육/훈련/자격증/배움 (education/training/certificates/learning)
   - Questions about government support or “내일배움카드”
   - Inquiries about acquiring specific skills or certifications

3. general (general conversation)
   - Simple greetings
   - Questions about system usage
   - Small talk or expressions of gratitude

Answer format:
{{
    "intent": "job|training|general",
    "confidence": 0.0~1.0,
    "explanation": "One line explaining the classification rationale"
}}

Special rules:
1. If there is any possibility of job-related context, classify as "job" (adjust confidence based on relevance).
2. If both job and training are mentioned, classify as "job" by priority.
3. If the intent is unclear but there is a potential for job seeking, classify as "job" with lower confidence.
4. If a job-seeking intent was present in previous conversation, consider subsequent related messages as "job."
5. If age, region, or job type is mentioned, it likely indicates "job."

Examples:
1. "서울에 일자리 있나요?" -> job (0.9)
2. "40대도 할 수 있나요?" -> job (0.8)
3. "안녕하세요" -> general (0.9)
4. "자격증 따고 싶어요" -> training (0.9)
5. "지역 근처에 뭐 있나요?" -> job (0.7)

""")

# 재랭킹 프롬프트 추가
rerank_prompt = PromptTemplate.from_template("""
Please compare the user's search criteria to each job posting and rate how well each posting matches.

User's criteria:
{user_conditions}

Job postings:
{documents}

Return the suitability score of each job posting as a JSON array from 0 to 5:
{{"scores": [score1, score2, ...]}}

Evaluation criteria:
- Exact region match: +2 points
- Exact job match: +2 points
- Matching age group: +1 point
- Nearby region: +1 point
- Similar job: +1 point

""")

# 훈련정보 관련 프롬프트 추가
TRAINING_PROMPT = PromptTemplate.from_template("""
You are a vocational training counselor for senior job seekers.
From the following user request, extract the information necessary to search for training programs.

User request: {query}

Please respond in the following JSON format:
{{
    "지역": "extracted region name",
    "과정명": "extracted training program name",
    "기간": "desired duration (if any)",
    "비용": "desired cost (if any)"
}}

Special rules:
1. If the region is not specified, leave it as an empty string.
2. If the training program name is not specified, leave it as an empty string.
3. The duration and cost are optional.

""")