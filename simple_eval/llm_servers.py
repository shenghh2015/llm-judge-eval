LLM_SERVERS = {
    "deepseek_r1_70b": "http://10.10.10.128:8000/v1",
    "deepseek_r1_32b": "http://10.10.10.121:8001/v1",
    "qwq_32b": "http://10.10.10.121:8000/v1",
    "qwen2_5_32b_instr": "http://10.10.10.121:8002/v1",
    "qwen2_5_72b_instr": "http://10.10.10.128:8001/v1",
    "llama3_1_70b_instr": "http://10.10.10.131:8000/v1",
    "llama3_3_70b_instr": "http://10.10.10.131:8001/v1",
    "gpt-4o": "https://api.openai.com/v1",
    "gpt-4o-mini": "https://api.openai.com/v1",
    "gpt-3.5-turbo": "https://api.openai.com/v1"
}

if __name__ == "__main__":
    
    from openai import OpenAI
    import os
    import time
    
    for model, base_url in llm_servers.items():
        print("*-" * 50)
        print(f"{model}: {base_url}")
        start_time = time.time()
        api_key = "EMPTY"
        if model in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
            api_key = os.environ.get("OPENAI_API_KEY")  # export
            base_url = "https://api.openai.com/v1"
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
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print()