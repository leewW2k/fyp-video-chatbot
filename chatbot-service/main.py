import openai_service
import dotenv

dotenv.load_dotenv()


def main():
    client = openai_service.OpenAIService()
    client.create_prompt_response()


if __name__ == "__main__":
    main()
