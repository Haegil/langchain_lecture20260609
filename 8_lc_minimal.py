from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# LLM 객체 생성 (LangChain에서 사용할 언어 모델)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# LLM 응답을 문자열로 변환하는 출력 파서
parser = StrOutputParser()

# 사용자 선택에 따라 사용할 프롬프트 템플릿 정의
prompt_map = {
    "1": ("요약", "다음 내용을 한 문장으로 요약해 주세요.\n내용: {text}"),
    "2": ("키워드", "다음 내용에서 핵심 키워드 5개만 뽑아 주세요.\n내용: {text}"),
    "3": ("답변", "다음 질문에 3문장 이내로 답변해 주세요.\n질문: {text}"),
}

# 처리 방식 선택 메뉴 출력
print("1) 요약")
print("2) 키워드")
print("3) 3문장 답변")
sel = input("선택(1~3): ").strip()

# 잘못된 선택 방지
if sel not in prompt_map:
    raise SystemExit("잘못된 선택")

# 선택된 프롬프트 템플릿 생성
name, template = prompt_map[sel]
prompt = PromptTemplate.from_template(template)

# LangChain 실행 파이프라인 구성
# PromptTemplate → LLM → OutputParser
chain = prompt | llm | parser

# 사용자 입력을 프롬프트 변수(text)에 바인딩하여 실행
text = input(f"[{name}] 입력: ").strip()
print(chain.invoke({"text": text}))
