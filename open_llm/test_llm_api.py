from openai import OpenAI
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-api", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="qwen25_7b_instr")
    return parser.parse_args()


def main():

    args = get_args()

    client = OpenAI(
        api_key="EMPTY",
        base_url=args.url_api,
    )

    prompt = "How many r's are in the word \"strawberry\""

    chat_response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    print("Chat response:", chat_response)

if __name__ == "__main__":
    main()
