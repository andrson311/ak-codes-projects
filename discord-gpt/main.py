import os
import json
import asyncio
import discord
from discord.ext import commands

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType, Tool


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

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)

class Chat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.hybrid_command(name='chat')
    async def chat(self, ctx, *, prompt):
        async with ctx.typing():
            response = agent.run(input=prompt)
        
        await ctx.send(f'<@{ctx.message.author.id}> says: {prompt}\n\n{response}')


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or('!'),
    description='A chatbot with access to the internet',
    intents=intents
)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('--------------------')
    await bot.tree.sync()

async def main():
    async with bot:
        await bot.add_cog(Chat(bot))
        await bot.start(get_token('discord-token'))

asyncio.run(main())
