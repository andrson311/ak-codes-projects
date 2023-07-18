import os
import json
import asyncio
import discord
from discord.ext import commands

ROOT = os.path.dirname(__file__)

def get_token(token_name):
    try:

        with open(os.path.join(ROOT, 'auth.json'), 'r') as auth_file:
            auth_data = json.load(auth_file)
            token = auth_data[token_name]
            return token
    except:
        return
    

class DemoCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.hybrid_command(name='embed')
    async def embed(self, ctx):
        embed = discord.Embed(
            color=discord.Color.red(),
            title='This is the title',
            description='This is a descriptions *first line*\nAnd this is the **second line**'
        )

        embed.set_footer(text='This is some text in the footer')
        embed.set_author(name='Andraž', url='https://ak-codes.com/')
        embed.set_thumbnail(url='https://ak-codes.com/wp-content/uploads/2023/05/cropped-logo.png')
        embed.set_image(url='https://ak-codes.com/wp-content/uploads/2023/06/20191103_113510-scaled.webp')
        embed.add_field(name='Blog', value='https://ak-codes.com/')

        await ctx.send(embed=embed)

    
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or('!'),
    description='Embed demonstration bot',
    intents=intents
)

@bot.event
async def on_ready():

    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    await bot.tree.sync()

async def main():
    async with bot:
        await bot.add_cog(DemoCog(bot))
        await bot.start(get_token('discord-token'))

asyncio.run(main())
