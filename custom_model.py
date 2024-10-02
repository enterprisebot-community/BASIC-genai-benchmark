import json
import os
import boto3
from botocore.exceptions import ClientError

import requests
from dotenv import load_dotenv


# Takes in the target model and messages to generate a custom response using a provider API
def get_custom_response(target_model, messages):
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

	if response.status_code != 200:
		print(f"ERROR:{response.json()}")
		return None, None, None

	answer = response.json()['choices'][0]['message']['content']

	prompt_tokens = response.json()['usage']['prompt_tokens']
	completion_tokens = response.json()['usage']['completion_tokens']

	return answer, prompt_tokens, completion_tokens


# Takes in the target model and messages to generate a custom response using AWS Bedrock
def get_bedrock_response(target_model, messages):
	load_dotenv()

	client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION"))

	# Convert messages to the required format
	conversation = [
		{
			"role": "user",
			"content": [{"text": msg["content"]} for msg in messages],
		}
	]

	try:
		# Send the message to the model, using a basic inference configuration.
		response = client.converse(
			modelId=target_model,
			messages=conversation,
			inferenceConfig={"maxTokens": 300, "stopSequences": ["User:"], "temperature": 0.9},
			additionalModelRequestFields={}
		)

		# Extract and print the response text.
		response_text = response["output"]["message"]["content"][0]["text"]
		input_tokens = response["usage"]["inputTokens"]
		output_tokens = response["usage"]["outputTokens"]

		return response_text, input_tokens, output_tokens

	except (ClientError, Exception) as e:
		print(f"ERROR: Can't invoke '{target_model}'. Reason: {e}")
		return None, None, None


if __name__ == "__main__":
	load_dotenv()

	# Test data
	target_model = "meta.llama3-8b-instruct-v1:0"
	messages = [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Tell me a joke."}
	]

	# Test get_bedrock_response
	bedrock_answer, bedrock_prompt_tokens, bedrock_completion_tokens = get_bedrock_response(target_model, messages)
	print("\nBedrock Response:")
	print("Answer:", bedrock_answer)
	print("Prompt Tokens:", bedrock_prompt_tokens)
	print("Completion Tokens:", bedrock_completion_tokens)
