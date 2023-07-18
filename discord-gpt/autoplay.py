import os
import json
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

ROOT = os.path.dirname(__file__)
def get_token(token_name):
    
    auth_file = open(os.path.join(ROOT, 'auth.json'))
    auth_data = json.load(auth_file)
    token = auth_data[token_name]
    return token

os.environ['OPENAI_API_KEY'] = get_token('openai-token')

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template='List 5 songs similar to {reference_song}.\n{format_instructions}',
    input_variables=['reference_song'],
    partial_variables={'format_instructions': format_instructions}
)

model = OpenAI(temperature=0.9)
inp = prompt.format(reference_song='Rise Against - Re-Education (Through Labor) (Uncensored)')
output = model(inp)

print(output_parser.parse(output))