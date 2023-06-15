import json
import os

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

ROOT = os.path.dirname(__file__)
def get_token(token_name):

    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

os.environ['OPENAI_API_KEY'] = get_token('openai-token')

chat = ChatOpenAI(temperature=0.9)
memory = ConversationBufferWindowMemory()
conversation = ConversationChain(
    llm=chat,
    memory=memory
)

while True:
    prompt = input('You: ')
    response = conversation.predict(input=prompt)

    print('AI:', response)
