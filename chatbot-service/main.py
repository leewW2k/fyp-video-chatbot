import openai_service
import dotenv
import asyncio

dotenv.load_dotenv()


async def main():
    client = openai_service.OpenAIService()
    await client.create_prompt_response()


if __name__ == "__main__":
    asyncio.run(main())
