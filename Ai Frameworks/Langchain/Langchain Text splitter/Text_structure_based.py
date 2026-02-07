#aslo called RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ LangChain is a framework for developing applications powered by language models.
It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
LangChain provides a standard interface for all LLMs, as well as a core set of modules
that can be used to build applications. This includes prompt management,
memory management, and integration with other data sources and APIs.
LangChain is designed to be modular and extensible, allowing developers to easily
customize and extend the framework to meet their specific needs."""

#initialize the RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    #separator = '\n'
)

chunks = splitter.split_text(text)
print('length of a chunk:' , len(chunks))   #prints the number of text chunks
print(chunks)   #prints the list of text chunks