from openai import OpenAI
import os


def main():

  # openai model: gpt-4o, gpt-to-mini
  # model = "gpt-4o-mini"
  # api_key = os.environ.get("OPENAI_API_KEY")  # export OPENAI_API_KEY=[YOUR_OPENAI_API_KEY]
  # base_url = ""

  # opensource model served by vllm
  # commond: vllm serve --model Qwen/Qwen2.5-7B-Instruct --port 8000 --served_model_name qwen25_7b_instr --tensor-parallel-size 1 --max-model-len 2048
  model = "qwen25_7b_instr"
  base_url = "http://localhost:8000/v1"
  api_key = "EMPTY"

  client = OpenAI(
      api_key=api_key,
      base_url=base_url,
  )

  prompt = "San Francisco is a"

  chat_response = client.chat.completions.create(
      model=model,
      messages=[
          {
              "role": "system",
              "content": "You are a helpful assistant."
          },
          {
              "role": "user",
              "content": prompt
          },
      ])
  print("Chat response:", chat_response)


if __name__ == "__main__":
  main()
