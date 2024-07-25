import os

import requests
from dotenv import load_dotenv


# gets llama response and returns answer + tokens used
def get_llama_response(target_model, messages):

	load_dotenv()

	stream = False
	url = os.environ.get("PROVIDER_API_URL")
	headers = {
		"Authorization": os.environ.get("PROVIDER_API_KEY"),
		"Content-Type": "application/json",
	}

	data = {
		"temperature": 0.9,
		"messages": messages,
		"model": target_model,
		"stream": stream,
		"frequency_penalty": 0.2,
		"max_tokens": 300
	}

	response = requests.post(url, headers=headers, json=data)

	# lets use choices.message.content

	if response.status_code != 200:
		print(f"ERROR:{response.json()}")
		return None, None, None

	answer = response.json()['choices'][0]['message']['content']

	prompt_tokens = response.json()['usage']['prompt_tokens']
	completion_tokens = response.json()['usage']['completion_tokens']

	return answer, prompt_tokens, completion_tokens
