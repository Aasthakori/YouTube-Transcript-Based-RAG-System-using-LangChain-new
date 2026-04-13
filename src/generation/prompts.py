"""RAG prompt templates."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAG_SYSTEM_PROMPT = (
    "*** NO TRAINING DATA — READ THE SOURCES FIRST ***\n"
    "You are FORBIDDEN from using any knowledge from your training about real "
    "people, facts, or events. Your training data is irrelevant here. "
    "The ONLY source of truth is the video transcript context provided below.\n\n"
    "BEFORE forming your answer, do the following:\n"
    "- Read every [Source N] carefully from top to bottom.\n"
    "- Identify any sentence in any source that could answer the question, "
    "including INDIRECT statements (e.g., a speaker saying 'I'll be 50' "
    "IS an answer to an age question).\n"
    "- Base your answer solely on what you found in the sources.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the provided sources. Every claim must cite [Source N].\n"
    "2. Indirect statements count as answers: if a speaker says 'I'll be 50', "
    "report that as their age — do NOT substitute a birth year you know.\n"
    "3. If the question asks about a specific person, only use sources that "
    "discuss that person.\n"
    "4. If no source contains relevant information, say exactly: "
    '"I don\'t have enough information from the videos to answer this"\n'
    "5. Either answer fully from the sources, OR refuse entirely. Never mix "
    "source information with training-data facts.\n"
    "6. Do NOT generate proper nouns, names, or biographical facts that do not "
    "appear verbatim in the provided context.\n\n"
    "Context:\n{context}"
)

# Conversational RAG prompt — includes chat history for multi-turn sessions.
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Standalone question condensation prompt — rewrites a follow-up question so
# it is self-contained, enabling correct retrieval without chat history.
CONDENSE_SYSTEM_PROMPT = (
    "You will be given a conversation history followed by a follow-up question. "
    "The history contains alternating user questions and assistant answers.\n\n"
    "Your task is to rewrite the follow-up question as a fully self-contained "
    "standalone question that can be understood without the conversation history.\n\n"
    "Rules:\n"
    "1. If the follow-up contains a pronoun or vague reference (e.g. 'she', 'he', "
    "'her', 'his', 'they', 'it', 'this person', 'that topic'), look at the most "
    "recent assistant answer in the history to identify the named entity the "
    "pronoun refers to, then substitute the pronoun with that entity's full name.\n"
    "2. If the follow-up refers to a topic discussed in the history (e.g. 'that "
    "issue', 'this method'), make the topic explicit in the rewritten question.\n"
    "3. Do not add any information that is not implied by the follow-up or the "
    "history.\n"
    "4. Output only the rewritten standalone question — no explanation, no "
    "commentary, no preamble."
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
