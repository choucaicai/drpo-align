

from openai import OpenAI


import os
from openai import OpenAI
import openai
import json

# api key
# api_key=  "sk-8jVyLPzsIyNMtepLnsmhPNqkm2ZEteHDMRMpj26741XawPds"
# api_url=  "https://api.chatanywhere.tech/v1"

api_key=  "sk-JDYgvFZFbMpM43Hy99B8Ee756dC3420fBf89633d6176Fb91"
api_url=  "https://api.uniapi.me/v1"

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

    import os
    import asyncio
    from openai import AsyncOpenAI


    async def ask_question(client, question):
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            )
            return question, response.choices[0].message.content
        except Exception as e:
            return question, f"Error: {str(e)}"

    async def process_questions(questions):
        # async with AsyncOpenAI(
        #     api_key=api_key,
        #     base_url="https://api.chatanywhere.tech/v1"
        # ) as client:
        async with AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        ) as client:        
            tasks = [ask_question(client, question) for question in questions]
            return await asyncio.gather(*tasks)

    async def main():
        questions = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote 'To Kill a Mockingbird'?",
            "What is the theory of relativity?",
            "How do you make a chocolate cake?"
        ]

        results = await process_questions(questions)

        for question, answer in results:
            print(f"Q: {question}")
            print(f"A: {answer}")
            print("-" * 50)

    asyncio.run(main())