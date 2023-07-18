import os
import json
import asyncio
import discord
from discord.ext import commands

ROOT = os.path.dirname(__file__)

# data directory
DATA = os.path.join(ROOT, 'data')

if not os.path.exists(DATA):
    os.mkdir(DATA)

def get_token(token_name):
    
    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

def save_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_data(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except:
        return False

role_message_id = 1123992117396635739
emoji_to_role = {
    discord.PartialEmoji(name='💰'): 1123995025819308103
}

intents = discord.Intents.default()
intents.members = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or('!'),
    description='Bot that manages roles.',
    intents=intents
)

def get_reaction_user_and_role(payload):
    guild = bot.get_guild(payload.guild_id)
    if guild is None:
        return
    
    try:
        role_id = emoji_to_role[payload.emoji]
    except KeyError:
        return
    
    role = guild.get_role(role_id)

    if role is None:
        return
    
    member = guild.get_member(payload.user_id)

    return member, role

@bot.event
async def on_raw_reaction_add(payload):
    if payload.message_id == role_message_id:
        member, role = get_reaction_user_and_role(payload)
        try:
            await member.add_roles(role)
        except discord.HTTPException:
            pass

@bot.event
async def on_raw_reaction_remove(payload):
    if payload.message_id == role_message_id:
        member, role = get_reaction_user_and_role(payload)
        try:
            await member.remove_roles(role)
        except discord.HTTPException:
            pass

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('----------')

async def main():
    async with bot:
        await bot.start(get_token('discord-token'))

asyncio.run(main())
