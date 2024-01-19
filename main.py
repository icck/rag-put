from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

texts = [
    "私の趣味は読書です",
    "私の好きな食べ物はラーメンです",
    "私の嫌いな食べ物はトマトです",
]

print("Indexing...")
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())
retriver = vectorstore.as_retriever(search_kwargs={"k": 1})
print("Indexed.")

prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけ基づいて回答して下さい。

    context: {context}

    質問: {question}
    """
)

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# RunnablePassthrough: 入力が retriever に渡されつつ、prompt にも渡される
# StrOutputParser: 出力が str に変換される
chain: Runnable = {"context": retriver, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

question = "私の趣味は何ですか？"
print("Invoking chain...")
print(f"question: {question}")
answer = chain.invoke(question)
print(f"answer: {answer}")
