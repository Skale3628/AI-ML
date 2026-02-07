from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model= 'mistral')

while True:
    user_input = input('you : ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print('AI :', result.content)
