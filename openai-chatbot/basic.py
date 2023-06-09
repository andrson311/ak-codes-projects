import json
import os
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

ROOT = os.path.dirname(__file__)
def get_token(token_name):

    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

os.environ['OPENAI_API_KEY'] = get_token('openai-token')

memory = ConversationBufferMemory()
llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

while True:
    prompt = input('You: ')
    if prompt:
        response = conversation.predict(input=prompt)
        print('Bot:', response)