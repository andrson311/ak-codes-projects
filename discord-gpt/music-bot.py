import os
import json
import asyncio
import discord
from discord.ext import commands
import yt_dlp as youtube_dl

ROOT = os.path.dirname(__file__)
def get_token(token_name):

    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

class MusicCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.is_playing = False
        self.is_paused = False

        self.music_queue = []
        self.load_queue()

        self.ydl_options = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(ROOT, 'yt', '%(extractor)s-%(id)s-%(title)s.%(ext)s'),
            'restrictfilenames': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'logtostderr': False,
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0',  # bind to ipv4 since ipv6 addresses cause issues sometimes
        }

        self.ffmpeg_options = {
            'options': '-vn'
        }

        self.voice_client = None
    
    def load_queue(self):
        try:
            with open(os.path.join(ROOT, 'queue.json'), 'r') as queue_file:
                self.music_queue = json.load(queue_file)
        except:
            print('Starting from empty queue.')

    def save_queue(self):
        with open(os.path.join(ROOT, 'queue.json'), 'w') as queue_file:
            json.dump(self.music_queue, queue_file, indent=4)
    
    def search_yt(self, item):
        with youtube_dl.YoutubeDL(self.ydl_options) as ydl:
            try:
                info = ydl.extract_info(item, download=True)
                
                if 'entries' in info:
                    info = info['entries'][0]
                    source = info['formats'][0]['url']
                else:
                    source = info['url']
                
                filename = ydl.prepare_filename(info)
            except:
                print('Something went wrong.')
    
        return {
            'source': source,
            'title': info['title'],
            'filename': filename
        }
    
    def play_next(self):
        if len(self.music_queue) > 0:
            self.is_playing = True
            filepath = self.music_queue[0]['filename']
            self.music_queue.pop(0)
            self.save_queue()

            self.voice_client.play(discord.FFmpegPCMAudio(filepath, **self.ffmpeg_options), 
                                   after=lambda e: self.play_next())
        else:
            self.is_playing = False

    async def play_music(self, ctx):
        if len(self.music_queue) > 0:
            self.is_playing = True
            channel = ctx.author.voice.channel

            filepath = self.music_queue[0]['filename']
            await ctx.send(f'Now playing: {self.music_queue[0]["title"]}')

            if self.voice_client == None or not self.voice_client.is_connected():
                self.voice_client = await channel.connect()

                if self.voice_client == None:
                    await ctx.send('Could not connect to the voice channel.')
                    return
            else:
                await self.voice_client.move_to(channel)
            
            self.music_queue.pop(0)
            self.save_queue()
            self.voice_client.play(discord.FFmpegPCMAudio(filepath, **self.ffmpeg_options),
                                   after=lambda e: self.play_next())
        else:
            self.is_playing = False

    
    @commands.hybrid_command(name='play')
    async def play(self, ctx, *, song):
        channel = ctx.author.voice.channel
        if channel is None:
            await ctx.send('You\'re not connected to a voice channel.')
        elif self.is_paused:
            self.voice_client.resume()
        else:
            async with ctx.typing():
                result = self.search_yt(song)
                if type(result) == type(True):
                    await ctx.send('Oops, something went wrong.')
                else:
                    self.music_queue.append(result)
                    self.save_queue()

                    if self.is_playing == False:
                        await self.play_music(ctx)
                    else:
                        await ctx.send(f'Added {result["title"]} to the queue.')

    @commands.hybrid_command(name='pause')
    async def pause(self, ctx):
        if self.is_playing:
            self.is_playing = False
            self.is_paused = True
            self.voice_client.pause()
        elif self.is_paused:
            self.is_paused = False
            self.is_playing = True
            self.voice_client.resume()
        
        await ctx.send('')
    
    @commands.hybrid_command(name='resume')
    async def resume(self, ctx):
        if self.is_paused:
            self.is_paused = False
            self.is_playing = True
            self.voice_client.resume()
        await ctx.send('')
    
    @commands.hybrid_command(name='skip')
    async def skip(self, ctx):
        if self.voice_client != None and self.voice_client:
            self.voice_client.stop()
            await self.play_music()
            await ctx.send('')
    
    @commands.hybrid_command(name='queue')
    async def queue(self, ctx):
        result = ''
        for q in self.music_queue:
            result += q['title'] + '\n'
        
        if result != '':
            await ctx.send(result)
        else:
            await ctx.send('Queue is empty.')
    
    @commands.hybrid_command(name='clear')
    async def clear(self, ctx):
        if self.voice_client != None and self.is_playing:
            self.voice_client.stop()
        self.music_queue = []
        self.save_queue()
        await ctx.send('Queue cleared.')
    
    @commands.hybrid_command(name='leave')
    async def leave(self, ctx):
        async with ctx.typing():
            self.is_playing = False
            self.is_paused = False
            await self.voice_client.disconnect()
    
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or('!'),
    description='A music bot.',
    intents=intents
)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')
    await bot.tree.sync()
    
async def main():
    async with bot:
        await bot.add_cog(MusicCog(bot))
        await bot.start(get_token('discord-token'))

asyncio.run(main())

