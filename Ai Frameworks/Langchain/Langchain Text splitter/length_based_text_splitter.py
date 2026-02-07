from langchain_text_splitters import CharacterTextSplitter  
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')   #specifying the path of the PDF file to be loaded
docs = loader.load()    #loads the document and returns a list of Document objects


#text = """LangChain is a framework for developing applications powered by language models. 
#It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."""

splitter = CharacterTextSplitter(
    chunk_size =100,
    chunk_overlap = 0,
    separator = ''
)

#result = splitter.split_text(text)   #splits the text into chunks based on the specified chunk size and overlap  
result = splitter.split_documents(docs)  #splits the Document objects into chunks based on the specified chunk size and overlap

print(result[0])   #prints the list of text chunks
