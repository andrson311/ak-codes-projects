import json
import os

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper

ROOT = os.path.dirname(__file__)
def get_token(token_name):

    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

os.environ['OPENAI_API_KEY'] = get_token('openai-token')
os.environ['SERPAPI_API_KEY'] = get_token('serpapi-token')

search = SerpAPIWrapper()
tools = [
    Tool(
        name='Current Search',
        func=search.run,
        description='useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term.'
    )
]

memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True)

llm = ChatOpenAI(temperature=0.9)
agent_chain = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory
)

while True:
    prompt = input('You: ')
    if prompt:
        response = agent_chain.run(input=prompt)

        print('AI:', response)

