import os
import pdb
import json
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import time
import logging

class GetResponse():
    # OpenAI model list: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    # Claude3 model list: https://www.anthropic.com/claude
    # "claude-3-opus-20240229", gpt-4-0125-preview
    def __init__(self, cache_file, model_name="gpt-4-0125-preview", max_tokens=1000, temperature=0):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_file = cache_file

        # invariant variables
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        # get model keys
        load_dotenv()
        if "gpt" in model_name:
            self.key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL")
            
            if not self.key:
                raise ValueError("API key not found in environment variables")

            self.client = OpenAI(api_key=self.key)
            self.seed = 1130 # TODO: set the seed configurable?
        elif "claude" in model_name:
            self.key = os.getenv("CLAUDE_API_KEY")
            self.client = Anthropic(api_key=self.key)
        # cache related
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 1
        self.print_interval = 20

    # Returns the response from the model given a system message and a prompt text.
    def get_response(self, system_message, prompt_text, cost_estimate_only=False, logprob_threshold=float('-inf')):
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if cost_estimate_only:
            # count tokens in prompt and response
            response_tokens = 0
            return None, None, prompt_tokens, response_tokens

        # check if prompt is in cache; if so, return from cache
        cache_key = prompt_text.strip()
        if cache_key in self.cache_dict:
            cached_response, cached_logprobs_data = self.cache_dict[cache_key]
            return cached_response, cached_logprobs_data, 0, 0

        # Example message: "You are a helpful assistant who can extract verifiable atomic claims from a piece of text."
        if "gpt" in self.model_name:
            message = [{"role": "system", "content": system_message},
                        {"role": "user", "content": prompt_text}]
            response_params = {
                                "model": self.model_name,
                                "messages": message,
                                "max_completion_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "seed": self.seed,
                                # "logprobs": True,
                                # "top_logprobs": 1  # Uncomment if you want to include this parameter
                            }
            # if no logprob_threshold is set: skip using the param logprobs because some models do not have the param
            if logprob_threshold != float('-inf'):
                response_params["logprobs"] = True
            try:
                response = self.client.chat.completions.create(**response_params)
                response_content = response.choices[0].message.content.strip()

                if "logprobs" in response_params:
                    # Convert logprobs to a serializable format (dict) to later save in cache
                    logprobs_data = {
                        'tokens': [token.token for token in response.choices[0].logprobs.content],
                        'logprob': [token.logprob for token in response.choices[0].logprobs.content],
                    } if response.choices[0].logprobs else None
                else:
                    logprobs_data = None
            except Exception as e:
                logging.error(f"Unexpected error: {e}. Skipping this sample.")
                return "", None, 0, 0

        elif "claude" in self.model_name:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_content = response.content[0].text.strip()
            logprobs_data = None

        # update cache
        self.cache_dict[cache_key] = (response_content.strip(), logprobs_data)
        self.add_n += 1

        # save cache every save_interval times
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        if self.add_n % self.print_interval == 0:
            print(f"Saving # {self.add_n} cache to {self.cache_file}...")
        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, logprobs_data, prompt_tokens, response_tokens

    # Returns the number of tokens in a text string.
    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
        else:
            cache = {}
        return cache
