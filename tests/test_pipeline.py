from langchain_core.messages import HumanMessage
from langchain_community.llms import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def test_full_conversational_chain_mock():
    mock_responses = [
        "What is garbage collection in Java?", 
        "Garbage collection in Java reclaims heap memory from unreachable objects."
    ]
    mock_llm = FakeListLLM(responses=mock_responses)

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase to standalone."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    rephrase_chain = rephrase_prompt | mock_llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer with context:\n{context}"),
        ("human", "{question}")
    ])
    qa_chain = qa_prompt | mock_llm | StrOutputParser()

    history = [HumanMessage(content="Explain Java memory management.")]
    standalone = rephrase_chain.invoke({"question": "How does GC work?", "chat_history": history})
    assert "garbage collection" in standalone.lower()

    output = qa_chain.invoke({"context": "GC runs in JVM heap.", "question": standalone})
    assert "heap memory" in output