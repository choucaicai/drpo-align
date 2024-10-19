

from openai import OpenAI


import os
from openai import OpenAI
import openai
import json

# api key
api_key=  "Your Openai Key"
api_url=  ""
rm_path = "/media/mil/LLMWeight/deberta-v3"


client = OpenAI(
    api_key=api_key,
    base_url=api_url
)

# test connection
if __name__ == "__main__":
    # chat_completion = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "how are you",
    #         }
    #     ],
    #     model="gpt-4",
    # )
    # print(chat_completion)
    pass
