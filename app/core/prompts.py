from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from app.utils.constants import DICTIONARY  
import re

# 사전 변환 규칙을 적용하는 함수 (DICTIONARY 직접 사용)
def apply_dictionary_rules(query: str) -> str:
    """사용자의 질문을 사전(DICTIONARY)에 따라 변환하는 함수"""
    pattern = re.compile("|".join(map(re.escape, DICTIONARY.keys())))
    return pattern.sub(lambda match: DICTIONARY[match.group(0)], query)


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

Search Guidelines:
1. Summarize key points from search results concisely
2. Include only essential information
3. Acknowledge if information is insufficient
4. You must always indicate the source. Use Markdown links for references.
5. Stick to verified facts
6. End with: '혹시 채용 정보나 직업 훈련에 대해 더 자세히 알아보고 싶으신가요?'
"""

# 기본 대화 프롬프트
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_persona_prompt),
    ("human", "{query}")  # input -> query로 변경하여 일관성 유지
])

# 정보 추출 프롬프트
EXTRACT_INFO_PROMPT = PromptTemplate.from_template("""
You are an expert at extracting job-related information from natural conversations.

Previous conversation context:
{chat_history}

Current message: {user_query}

Task: Extract job type, region, and age group information from the conversation.
Use the previous conversation context to supplement any missing information.

Common Expression References:
1. Job Type Keywords:
   - Direct: 일자리, 자리, 일거리, 직장
   - Actions: 취직, 취업
   
2. Location Keywords:
   - Administrative districts: 서울특별시, 서울시, 서울, 강남구, 강북구 등
   - Only extract actual district names, not relative locations
   - If user mentions relative locations (여기, 이 근처, 우리 동네 등), leave location empty
   
3. Age Group Keywords:
   - Senior terms: 시니어, 노인, 어르신, 중장년
   - Should be standardized to "시니어" in output

Output Format:
{{
    "직무": "extracted job type or empty string",
    "지역": "extracted region or empty string",
    "연령대": "extracted age group or empty string"
}}

Extraction Rules:
1. For non-specific job mentions (일자리, 일거리, 자리), use empty string for job type
2. Only extract actual administrative district names for location
3. If location is relative (여기, 근처 등), leave location field empty
4. Standardize all senior-related terms to "시니어"
5. Use context from previous conversation when relevant

Examples:
1. "서울에서 경비 일자리 좀 알아보려고요" -> {{"직무": "경비", "지역": "서울", "연령대": ""}}
2. "우리 동네 근처에서 할만한 일자리 있나요?" -> {{"직무": "", "지역": "", "연령대": ""}}
3. "강남구에서 요양보호사 자리 있을까요?" -> {{"직무": "요양보호사", "지역": "강남구", "연령대": ""}}
4. "여기 근처 식당 일자리 있나요?" -> {{"직무": "식당", "지역": "", "연령대": ""}}
""")

# 의도 분류 프롬프트 수정
CLASSIFY_INTENT_PROMPT = PromptTemplate.from_template("""
You are an expert career counselor specializing in senior job seekers. Your task is to accurately identify the user's intent, particularly focusing on job search or vocational training intentions.

Previous conversation:
{chat_history}

Current message: {user_query}

Intent Categories:
1. job (Job Search Related)
   - Contains keywords: 일자리, 직장, 취업, 채용, 자리
   - Location or position mentions (e.g., "Seoul", "경비", "요양보호사")
   - Age/experience/job requirements
   - Salary or working hours inquiries
   - Any expression of job seeking

2. training (Vocational Training Related)
   - Keywords: 교육, 훈련, 자격증, 배움
   - Government support or "내일배움카드" inquiries
   - Questions about skill acquisition or certification

3. general (General Conversation)
   - Greetings
   - System usage questions
   - Small talk or gratitude expressions

Response Format:
{{
    "intent": "job|training|general",
    "confidence": 0.0~1.0,
    "explanation": "One line explaining the classification rationale"
}}

Classification Rules:
1. Prioritize "job" intent if there's any job-related context
2. If both job and training are mentioned, classify as "job"
3. For unclear intents with potential job seeking, use "job" with lower confidence
4. Consider previous job-seeking context for subsequent messages
5. Age, location, or job type mentions likely indicate "job" intent

Examples:
1. "서울에 일자리 있나요?" -> {{"intent": "job", "confidence": 0.9, "explanation": "Direct job search request with location"}}
2. "40대도 할 수 있나요?" -> {{"intent": "job", "confidence": 0.8, "explanation": "Age-related job inquiry"}}
3. "안녕하세요" -> {{"intent": "general", "confidence": 0.9, "explanation": "Simple greeting"}}
4. "자격증 따고 싶어요" -> {{"intent": "training", "confidence": 0.9, "explanation": "Certificate acquisition inquiry"}}
5. "지역 근처에 뭐 있나요?" -> {{"intent": "job", "confidence": 0.7, "explanation": "Implicit job search with location"}}
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
NER_TRAINING_PROMPT = PromptTemplate.from_template("""
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

TRAINING_EXPLANATION_PROMPT = ChatPromptTemplate.from_template("""
You are a professional vocational training counselor. Please analyze and explain the following training courses from a professional perspective.

Training Courses:
{courses}

Please include the following in your explanation:
1. Key features and advantages of each course
2. Employment prospects and career paths
3. Prerequisites and preparation requirements
4. Cost-effectiveness analysis
5. Assessment of training duration and methods

Response Format:
- Maintain a professional and objective tone
- Provide clear and specific information
- Include realistic yet encouraging advice
- Use easily understandable terminology
- Focus on practical benefits for senior job seekers

Special Considerations:
- Highlight courses with high employment rates
- Explain government support or subsidies if available
- Mention any age-friendly features
- Address common concerns of senior learners
- Suggest preparation steps for successful completion

Structure your response as:
1. Overview of available courses
2. Detailed analysis of each course
3. Practical recommendations
4. Next steps for enrollment
                                                               
Always provide your answer in Korean.
""")