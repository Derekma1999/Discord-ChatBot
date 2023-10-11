import discord, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Get user input
    user_input = message.content

    # Generate dynamic response using GPT-2 model
    response = generate_gpt2_response(user_input)

    # Send the response back to the user
    await message.channel.send(response)

def generate_gpt2_response(user_input):
    # Tokenize the user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate response using GPT-2 model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1,  attention_mask=input_ids.ne(1))
    
    # Decode the generated response and remove special tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Replace 'YOUR_DISCORD_BOT_TOKEN' with your actual bot token
client.run('YOUR_DISCORD_BOT_TOKEN')