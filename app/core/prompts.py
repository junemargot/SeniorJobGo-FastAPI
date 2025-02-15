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
   - Relative: 여기, 이 근처, 우리 동네, 근처, 가까운
   - Should be standardized to "근처" in output
   
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
2. Standardize all proximity terms (여기, 이 근처, etc.) to "근처"
3. Standardize all senior-related terms to "시니어"
4. Use context from previous conversation when relevant

Examples:
1. "서울에서 경비 일자리 좀 알아보려고요" -> {{"직무": "경비", "지역": "서울", "연령대": ""}}
2. "우리 동네 근처에서 할만한 일자리 있나요?" -> {{"직무": "", "지역": "근처", "연령대": ""}}
3. "시니어가 할 만한 요양보호사 자리 있을까요?" -> {{"직무": "요양보호사", "지역": "", "연령대": "시니어"}}
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

TRAINING_EXTRACT_PROMPT = PromptTemplate.from_template("""
Extract training/education-related information from the user's message.

Training Classification Reference:
1. Training Types (훈련종류):
   - National Tomorrow Learning Card (국민내일배움카드훈련)
   - Business Training (사업주훈련)
   - Consortium Training (컨소시엄훈련)
   - Work and Learning (일학습병행)
   - Unemployed Training (실업자훈련)
   - Employee Training (재직자훈련)

2. Training Fields (훈련분야):
   - IT/Development: AI, Artificial Intelligence, Programming, Big Data, Cloud
   - Office: Management, Accounting, Marketing, HR
   - Service: Care Worker, Cooking, Beauty
   - Technical: Machinery, Electrical, Construction, Automotive

3. Training Locations (훈련지역):
   - Cities: Seoul, Gyeonggi, Incheon, Busan, Daegu, etc.
   - Seoul Districts: Gangnam-gu, Gangdong-gu, Nowon-gu, etc.

4. Training Methods (훈련방법):
   - In-person Training (집체훈련)
   - On-site Training (현장훈련)
   - Remote Training (원격훈련)
   - Blended Training (혼합훈련)

User Message: {user_query}

Please extract and return the following information in JSON format:
{{
    "training_type": "Training type in Korean (empty if none)",
    "training_field": "Training field keyword in Korean (empty if none)",
    "location": "Location in Korean (empty if none)",
    "training_method": "Training method in Korean (empty if none)",
    "cost_info": "Cost-related information in Korean (if any)"
}}

Examples:
1. Input: "AI 관련 온라인 강의 찾아줘"
   Output: {{
       "training_field": "AI",
       "training_method": "원격훈련",
       "location": "",
       "training_type": "",
       "cost_info": ""
   }}

2. Input: "서울 강남구에서 국민내일배움카드로 들을 수 있는 프로그래밍 수업 알려줘"
   Output: {{
       "training_field": "프로그래밍",
       "location": "서울 강남구",
       "training_type": "국민내일배움카드훈련",
       "training_method": "",
       "cost_info": ""
   }}

Important Notes:
1. Always return Korean text in the output JSON
2. Match training types and methods exactly as specified in the reference
3. For locations, maintain the exact district names (e.g., "강남구" not just "강남")
4. Keep field values empty ("") if not explicitly mentioned in the user message
""")

# 이력서 작성 가이드 프롬프트 추가
RESUME_GUIDE_PROMPT = PromptTemplate.from_template("""
You are a professional career counselor specializing in helping senior job seekers write effective resumes.

User Query: {query}
Previous Chat History: {chat_history}

Task: Provide tailored resume writing advice based on the user's specific question or needs.

Guidelines for Response:
1. Basic Information Section
   - Contact details (phone, email)
   - Professional photo guidelines
   - Address format

2. Work Experience Section
   - Reverse chronological order
   - Achievement-focused descriptions
   - Quantifiable results
   - Senior-friendly job history presentation

3. Education & Certifications
   - Relevant certifications first
   - Recent training or courses
   - Skills development emphasis

4. Core Competencies
   - Age-advantage skills
   - Transferable skills
   - Industry-specific expertise
   - Technology proficiency level

5. Self-Introduction
   - Experience highlights
   - Motivation statement
   - Value proposition
   - Career transition explanation (if applicable)

Special Considerations for Senior Job Seekers:
1. Focus on recent experience (last 10-15 years)
2. Emphasize adaptability and learning ability
3. Highlight wisdom and stability
4. Address technology comfort level honestly
5. Showcase mentoring/leadership abilities

Format your response:
1. Keep it concise and clear
2. Use bullet points for easy reading
3. Provide specific examples
4. Include age-appropriate language
5. Focus on strengths relevant to the target position

Remember:
- Be encouraging and supportive
- Emphasize experience as an advantage
- Provide practical, actionable advice
- Address age-related concerns professionally

Response should be structured as:
1. Direct answer to the specific question
2. Relevant examples or templates
3. Additional tips specific to senior job seekers
4. Next steps or follow-up suggestions
""")

# 이력서 피드백 프롬프트 추가
RESUME_FEEDBACK_PROMPT = PromptTemplate.from_template("""
You are a professional resume reviewer specializing in senior job seeker resumes.

Resume Content: {resume_content}
Job Target: {job_target}

Task: Provide constructive feedback on the resume with special consideration for senior job seekers.

Analysis Areas:
1. Overall Presentation
   - Layout and formatting
   - Length and conciseness
   - Professional appearance

2. Content Effectiveness
   - Relevance to target position
   - Achievement highlighting
   - Experience presentation
   - Skills emphasis

3. Age-Smart Strategies
   - Recent experience focus
   - Technology skills presentation
   - Adaptability demonstration
   - Wisdom/experience leverage

4. Red Flags
   - Age discrimination triggers
   - Outdated information
   - Gaps in employment
   - Technical skill gaps

Provide feedback in the following format:
1. Strengths (3-4 points)
2. Areas for Improvement (3-4 points)
3. Specific Recommendations
4. Additional Resources or Next Steps

Remember:
- Be constructive and encouraging
- Focus on actionable improvements
- Consider industry-specific needs
- Address age-related concerns tactfully
""")