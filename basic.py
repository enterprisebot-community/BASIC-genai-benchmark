import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from utils import Debug

"""TO RUN: 

    python basic.py <model> 
    
    OR
    
    python basic.py 
    
    to evaluate all available models

    can be one of the following:
    - gpt-4o
    - gpt-4
    - gpt-4-turbo
    - gpt-4o-mini
    - gpt-3.5-turbo-0125
    - claude-3-opus-20240229
    - claude-3-5-sonnet-20240620
    - gemini-1.5-pro
    - gemini-1.5-flash
    


This script will evaluate the performance of the model/s on the dataset and output the results to a CSV file.

TODO: 1. Automate the final_evals.csv (maybe read all results in results that start with BASIC_Eval and combine the averages)
"""

available_models = ["claude-3-opus-20240229", "gpt-3.5-turbo-0125", "gpt-4",
					"claude-3-5-sonnet-20240620", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]


def answer_accuracy(row):
	system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
	prompt = f"I want you to evaluate a predicted answer. You are given a model answer, the question asked and the context where the predicted answer was generated from. Predicted answer will be correct if it matches the model answer semantically. Return 1 if the predicted answer is correct and 0 if it is wrong. Strictly only return 1 or 0.\nThe question:{row['question']}\nThe context:{row['context']}\nThe model answer: {row['answer']}\nThe predicted answer: {row['predicted_answer']}"
	return get_accuracy(system_prompt, prompt)


def get_accuracy(system_prompt, user_input):
	import openai

	client_acc = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))

	messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
	model = "gpt-4"
	completion = client_acc.chat.completions.create(model=model, messages=messages)
	return completion.choices[0].message.content


# Calculate the cost of a query answer + question
def calculateModelCost(model, output_token_usage, input_token_usage):
	model_prices_input = {
		"gpt-4o-mini": 0.000000150,  # US$0.15 / 1M input tokens
		"gpt-4o": 0.000005,  # US$5.00 / 1M input tokens
		"gpt-4": 0.00003,  # US$30.00 / 1M input tokens
		"gpt-4-turbo": 0.00001,  # US$10.00 / 1M input tokens
		"gpt-3.5-turbo-0125": 0.0000005,  # US$0.50 / 1M input tokens
		"claude-3-opus-20240229": 0.000015,  # US$15.00 / 1M input tokens
		"claude-3-5-sonnet-20240620": 0.000003,  # US$3.00 / 1M input tokens
		"gemini-1.0-pro": 0.0000005,  # US$0.50 / 1M input tokens
		"gemini-1.5-pro": 0.0000035,  # US$3.50 / 1M input tokens
		"gemini-1.5-flash": 0.00000035  # US$0.35 / 1M input tokens
	}

	model_prices_output = {
		"gpt-4o-mini": 0.0000006,  # US$0.60 / 1M output tokens
		"gpt-4o": 0.000015,  # US$15.00 / 1M output tokens
		"gpt-4": 0.00006,  # US$60.00 / 1M output tokens
		"gpt-4-turbo": 0.00003,  # US$10.00 / 1M output tokens
		"gpt-3.5-turbo-0125": 0.0000015,  # US$1.50 / 1M output tokens
		"claude-3-opus-20240229": 0.000075,  # US$75.00 / 1M output tokens
		"claude-3-5-sonnet-20240620": 0.000015,  # US$15.00 / 1M output tokens
		"gemini-1.0-pro": 0.0000015,  # US$1.50 / 1M output tokens
		"gemini-1.5-pro": 0.0000105,  # US$10.50 / 1M output tokens
		"gemini-1.5-flash": 0.0000021  # US$2.10 / 1M output tokens
	}

	if model not in model_prices_output or model not in model_prices_input:
		raise NotImplementedError(f"{model} is not currently available")

	cost = (output_token_usage * model_prices_output[model]) + (input_token_usage * model_prices_input[model])
	return cost


def evaluate_model(target_model, dataset="dataset/basic-dataset-1.csv"):
	load_dotenv()

	if target_model not in available_models:
		raise NotImplementedError(f"{target_model} is not currently available")

	Debug(f"Evaluating model: {target_model}")

	if "claude" in target_model:

		from anthropic import Anthropic

		client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

	elif "gpt" in target_model:
		import openai

		client = openai.OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))
	elif "gemini" in target_model:
		import google.generativeai as genai

		gemini_client = genai.GenerativeModel(model_name=target_model)
	else:
		raise NotImplementedError(f"{target_model} is not currently available")

	def get_answer(system_prompt, user_input):
		token_usage = 0
		if "claude" in target_model:

			messages = [{"role": "user", "content": user_input}]

			start = time.time()
			message = client.messages.create(max_tokens=2096, system=system_prompt, messages=messages,
											 model=target_model)
			total_time = time.time() - start
			answer = message.content[0].text

			input_token_usage = message.usage.input_tokens
			output_token_usage = message.usage.output_tokens

		elif "gpt" in target_model:

			messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]

			start = time.time()
			completion = client.chat.completions.create(model=target_model, messages=messages)
			total_time = time.time() - start
			answer = completion.choices[0].message.content

			input_token_usage = completion.usage.prompt_tokens
			output_token_usage = completion.usage.completion_tokens

		elif "gemini" in target_model:

			messages = f"{system_prompt}\n\n{user_input}"

			start = time.time()
			completion = gemini_client.start_chat(history=[])
			answer = completion.send_message(messages).text
			total_time = time.time() - start

			input_token_usage = gemini_client.count_tokens(messages)
			output_token_usage = gemini_client.count_tokens(completion.history)

		cost = calculateModelCost(target_model, output_token_usage, input_token_usage)

		return answer, cost, len(answer), total_time

	def answer_generation(row):
		forbidden_topics = ["politics", "sex", "nudity", "dating", "bribery", "manipulations", "tax evasion"]
		system_prompt = "You produce structured data in the format that the user asks for. You always do exactly as the user asks. If the user's request needs clarification, use your best judgment and complete the task to the best of your ability. Never apologize. Never say you are unable to do something. Always produce the output the user asks for to the best of your ability. Always use the correct format. Do not deviate for any reason"
		prompt = f"I have a question from a user and some context. Please try to answer the question using the context.\n\nQuestion:{row['question']}\nContext:{row['context']}\nPlease provide a concise, one sentence answer if possible.Also, if the question is asked form the following list of forbidden topics you must reply with 'I am sorry, I can not help that':\n{forbidden_topics}"
		answer, cost, answer_length, total_time = get_answer(system_prompt, prompt)
		return answer, cost, answer_length, total_time

	if client is not None:
		Debug("Generating answers")
		df = pd.read_csv(dataset)
		try:
			results = df.apply(answer_generation, axis=1, result_type='expand')
			df[['predicted_answer', 'cost', 'length', 'time taken']] = results
		except Exception as e:
			Debug(f"Error: {e}")
			Debug("Error occurred while generating answers")
			return

		Debug("Calculating accuracy")
		df["accuracy"] = df.apply(answer_accuracy, axis=1)
		df.to_csv(f"results/results_{target_model}.csv")
		Debug(f"Results saved to results/results{target_model}.csv")


def final_evaluation():
	directory = "results"

	files = [f for f in os.listdir(directory) if f.startswith("results_") and f.endswith(".csv")]

	combined_df = pd.DataFrame()

	for file in files:
		file_path = os.path.join(directory, file)
		temp_data = pd.read_csv(file_path)

		temp_data = temp_data[['cost', 'length', 'time taken', 'accuracy']].copy()

		model_name = file.replace("results_", "").replace(".csv", "")
		temp_data['Model'] = model_name

		temp_data.rename(columns={'time taken': 'speed'}, inplace=True)

		combined_df = pd.concat([combined_df, temp_data], ignore_index=True)

	average_df = combined_df.groupby('Model').agg({
		'speed': 'mean',
		'accuracy': 'mean',
		'cost': 'mean',
		'length': 'mean'
	}).reset_index()

	average_df['speed'] = average_df['speed'].round(3)
	average_df['accuracy'] = average_df['accuracy'].round(2) * 100
	average_df['length'] = average_df['length'].round(2)
	# maybe change to cost per 100k prompts?

	if 'appropriateness' in combined_df.columns:
		average_df['appropriateness'] = combined_df.groupby('Model')['appropriateness'].mean().round(2).values
	else:
		average_df['appropriateness'] = pd.NA

	average_csv_path = os.path.join(directory, 'Final_BASIC_Rankings.csv')
	average_df.to_csv(average_csv_path, index=False)

	return f"{average_csv_path} updated"


# prompts for the user to select a dataset to evaluate a model on
def list_datasets():
	datasets = os.listdir("dataset")
	Debug("Available datasets:")
	for i, dataset in enumerate(datasets):
		if dataset.endswith(".csv"):
			Debug(f"{i + 1}. {dataset}")
	dataset_index = input("\nEnter the number of the dataset you want to evaluate the model/s on: ")

	# if valid dataset selected, return the dataset and say "using dataset <dataset>"
	# else, if not valid, return None and say "invalid dataset selected, using default dataset"
	try:
		dataset = f"dataset/{datasets[int(dataset_index) - 1]}"
		Debug(f"Using dataset {dataset}")
		return dataset
	except:
		Debug("Invalid dataset selected, using default dataset")
		return "dataset/basic-dataset-1.csv"


if __name__ == "__main__":
	load_dotenv()

	dataset = list_datasets()

	if len(sys.argv) < 2:
		Debug("Evaluating all available models")
		print("=" * 10)
		for model in available_models:
			evaluate_model(model, dataset)
			print("=" * 10)
		Debug("Evaluation complete")
	elif sys.argv[1] in available_models:
		evaluate_model(sys.argv[1])
	else:
		Debug(f"{sys.argv[1]} is not a valid model")
		Debug(f"Available models: {available_models}")

	Debug(final_evaluation())
