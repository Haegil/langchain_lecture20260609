import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)


@tool
def calc(expression: str) -> str:
    """
    사칙연산 계산 도구입니다.
    입력 예: 12*3, (100-20)/4
    """
    expr = expression.strip()

    if not re.fullmatch(r"[0-9.\s+\-*/()]+", expr):
        return (
            "허용되지 않은 수식입니다. "
            "숫자와 + - * / ( ) . 만 사용합니다."
        )

    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def local_search(keyword: str) -> str:
    """
    로컬 문서(data/notes.txt)에서
    키워드를 포함하는 문장을 찾아 반환합니다.
    """
    p = Path("data/notes.txt")

    if not p.exists():
        return "data/notes.txt 파일이 없습니다."

    text = p.read_text(encoding="utf-8")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    k = keyword.strip().lower()

    hits = [
        line
        for line in lines
        if k in line.lower()
    ]

    if not hits:
        return "검색 결과가 없습니다."

    return "검색 결과:\n" + "\n".join(
        f"- {h}" for h in hits
    )


tools = [calc, local_search]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
당신은 실습용 AI Agent입니다.

필요할 때만 도구를 사용합니다.

계산이 필요하면 calc를 사용합니다.
로컬 문서 근거가 필요하면 local_search를 사용합니다.

도구 결과를 반영하여
최종 답변은 4문장 이내로 작성합니다.
"""
)

print("AI Agent 미니 실습 시작 (/quit 입력 시 종료)")

while True:
    user_input = input("User> ").strip()

    if user_input == "/quit":
        break

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    print("\nAI>")

    messages = result["messages"]

    print(messages[-1].content)
    print()