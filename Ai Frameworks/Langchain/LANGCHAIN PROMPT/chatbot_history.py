from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model= 'mistral')

chat_history = []

while True:
    user_input = input('you : ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    chat_history.append(result.content)
    print('AI :', result.content)

print(chat_history)
