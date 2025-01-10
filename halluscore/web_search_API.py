'''
Using s.jina.ai for web search. See jina reader website and repo for more information.

Parameters in the script:
- max_workers
- max_retries

retry mechanism:
1. retriable errors: (HTTP Status Codes:)
    524 (Timeout): The server took too long to respond. This might happen if the service is under heavy load or experiencing a temporary issue.
    503 (Service Unavailable): The server is temporarily unable to handle the request. This could be due to maintenance or overload.
    429 (Too Many Requests): The server is rate-limiting you. Retrying after some time (respecting the Retry-After header, if present) is usually recommended.
2. non-retriable error:
    those on the client or sever side which a retry cannot fix. e.g., :
    400 (Bad Request): The request is malformed (e.g., invalid URL or parameters). Fix the query or input.
    401 (Unauthorized): The API key or credentials are missing or invalid. Check authentication details.
    403 (Forbidden): Access is denied, possibly due to lack of permissions or exceeding quotas. Verify your access rights or contact support.
    404 (Not Found): The requested resource doesn't exist. Verify the endpoint or query.
    500 (Internal Server Error): Indicates a problem with the server itself. While this might resolve with a retry, it often requires contacting support or waiting for a server fix.
'''
# TODO: clean code

import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import time


class WebSearchAPI():
    def __init__(self):
        # load environment variables
        load_dotenv()
        self.jina_key = os.getenv("JINA_KEY")
        # print(f'Using Jina Key: {self.jina_key} to search.')
        
        # define URL and headers for the API request
        self.url_prefix = 'https://s.jina.ai/'
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.jina_key}',
            'X-Token-Budget': '20000',
            'X-Remove-Selector': 'header, .class, #id', # remove page headers, etc
            'X-Retain-Images': 'none',
            'X-Return-Format': 'text', # return only text, no links or html markdown embedded in the output (the downside is the output also does not show the url that the text content is retrieved from)
            'X-Timeout': '100',
            'X-With-Generated-Alt': 'true'
        }
        
        # innitialize cache related variables
        self.cache_file = "data/cache/web_search_cache.json"
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 10

        # set parameters for retries and multiprocessing
        self.max_retries = 10
        self.max_workers = 5
        self.lock = threading.Lock()  # For thread-safe cache operations

    def encode_url(self, query, search_res_num):
        encoded_query = quote(query)
        url = self.url_prefix + f'{encoded_query}count={search_res_num}'
        return url

    def get_content(self, claim_lst, search_res_num):
        '''
        Return search_content_dict by search for search_res_num evidences for each claim in claim_lst.
        search_content_dict:
            {
             'claim1' : [{'title': ..., 'description': ..., 'url': ..., 'content': ..., 'usage': {"token": x}}, {...}, ...], 
             'claim2' : [{}, {}, {}, ...], 
             'claim3' : [...],
             ...
             }
        '''
        search_content_dict = {}
        # Use ThreadPoolExecutor to search queries in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:  # PRM is 40 for Jina premium, 20 for Jina free.
            # Submit tasks for each query
            future_to_query = {
                executor.submit(self.get_search_res, query, search_res_num): query
                for query in claim_lst
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_query), total=len(claim_lst), desc="Fetching web content"):
                query = future_to_query[future]
                try:
                    content_res = future.result()
                    search_content_dict[query] = content_res.get('data')
                except Exception as e:
                    print(f"Error processing query '{query}': {e}")
        return search_content_dict


    def get_search_res(self, query, search_res_num):
        '''
        Get search_res_num search results from API for a claim (query).
        '''
        # Thread-safe cache check
        cache_key = query.strip()
        with self.lock:
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key]

        # search in max_retries
        retries = 0
        while retries < self.max_retries:
            try:
                # Encode and send the request
                encoded_url = self.encode_url(query, search_res_num)
                response = requests.get(encoded_url, headers=self.headers, timeout=15)

                # Success
                if response.status_code == 200:
                    content_json = response.json()

                    with self.lock:
                        # Update cache
                        self.cache_dict[cache_key] = content_json
                        self.add_n += 1
                        # Save cache periodically
                        if self.add_n % self.save_interval == 0:
                            self.save_cache()
                    return content_json
                
                # Error
                elif response.status_code in [503, 524]:
                    print(f"Retryable error for query '{query}' (status: {response.status_code}). Retrying...")
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"Rate limit hit for '{query}'. Retrying after {retry_after} seconds...")
                    backoff_time = retry_after
                else:
                    print(f"Non-retryable error for query '{query}': {response.status_code}")
                    break
            except requests.exceptions.Timeout:
                print(f"Timeout error for query '{query}'. Retrying...")
            except requests.exceptions.RequestException as e:
                print(f"Request exception for query '{query}': {e}")
                break
                
            # Increment retries and apply exponential backoff with jitter
            retries += 1
            backoff_time = random.uniform(0, min(2 ** retries, 30))
            time.sleep(backoff_time)

        # Log and cache failure
        print(f"Max retries exceeded for '{query}'. Returning empty result.")
        with self.lock:
            self.cache_dict[cache_key] = {'data': []}
        return {'data': []}
        

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        # print(f"Saving web search cache ...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
                # print(f"Loading cache ...")
        else:
            cache = {}
        return cache
