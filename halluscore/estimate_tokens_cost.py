import tiktoken
import os
import regex
import pdb
import json
import spacy
from tqdm import tqdm
import argparse
from .claim_extractor import ClaimExtractor

class TokenEstimator():
    def __init__(self, model_name="gpt-4o", max_tokens=1000, temperature=0):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize the tokenizer based on the model
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Estimates the number of tokens in the prompt and the response
    def estimate_tokens(self, system_message, prompt_text):
        # Count tokens in the system message and the prompt text
        system_tokens = len(self.tokenizer.encode(system_message))
        prompt_tokens = len(self.tokenizer.encode(prompt_text))

        # Estimate the number of response tokens (based on the max_tokens and not an actual API call)
        response_tokens = self.max_tokens

        # Total tokens used (prompt + system message + estimated response)
        total_tokens = system_tokens + prompt_tokens + response_tokens
        return prompt_tokens, response_tokens, total_tokens

    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

class PromptFormatter():
    def __init__(self):
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.demo_path = os.path.join('data/demos/', 'few_shot_examples.jsonl')
        self.label_n = 3 # TODO
        self.model_name = 'gpt-4o'
    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]
    
    def get_prompt_template(self, qa_input):
        if qa_input:
            prompt_template = open("./prompt/extraction_qa_template.txt", "r").read()
        else:
            prompt_template = open("./prompt/extraction_non_qa_template.txt", "r").read()

        return prompt_template
    def get_verify_prompt_template(self):
        # get template
        prompt_template = open("./prompt/verification_instruction_trinary.txt", "r").read()

        # fill few-shot demo
        with open(self.demo_path, "r") as f:
            example_data = [json.loads(line) for line in f if line.strip()]
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            human_label = dict_item["human_label"]
            if self.label_n == 2:
                if human_label == "support":
                    human_label = "Supported."
                else:
                    human_label = "Unsupported."
            if "claude" in self.model_name:
                element_lst.extend([search_result_str, claim, human_label])
            else:
                element_lst.extend([claim, search_result_str, human_label])

        prompt_few_shot = prompt_template.format(*element_lst)
        return prompt_few_shot

    def sliding_window_prompt(self, question, response):
        """
        Given a model output to a question
        - split the response into sentences using spaCy
        - snippet = question (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - return snippet list (prompt list)
        """
        sentences = self.get_sentence(response)
        # new return values
        snippet_lst = []
        for i, sentence in enumerate(sentences):
            context1 = " ".join(sentences[max(0, i - 3):i])
            sentence = f"<SOS>{sentences[i].strip()}<EOS>"
            context2 = " ".join(sentences[i + 1:i + 2])

            snippet = f"Question: {question.strip()}\nResponse: {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            # new return value
            snippet_lst.append(snippet)
        return sentences, snippet_lst
    
    def chunk_prompt(self, question, response, stride):
        """
        Given a model output to a question
        - return snippet list (chunk prompt list)

        stride means the sentence count of a chunk. 
        So you have the options to set:
            a fixed stride (stride = 0 means whole response no stride; other positive intergers mean the actual stride) 
            or a dynamic one based on response length (stride = -1).
        """
        chunks = ClaimExtractor.get_chunk(response, stride)

        for i, chunk in enumerate(chunks):
            snippet = f"Question: {question.strip()}\nResponse: <SOS>{chunk.strip()}<EOS>".strip()
            # new return value
            snippet_lst.append(snippet)

        return sentences, snippet_lst


# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--extraction_method", type=str, required=True, choices=['chunk', 'sliding_window'])
    parser.add_argument("--stride", type=int, default=0, help='0 means feeding the whole response for extraction; -1 means dynamic stride based on response length.', required=False)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--search_res_num", type=int, default=10)
    args = parser.parse_args()

    estimator = TokenEstimator(max_tokens=1000)
    prompt_formatter = PromptFormatter()
    total_prompt_tok_cnt, total_resp_tok_cnt = 0, 0
    total_verify_prompt_tok_cnt, total_verify_resp_tok_cnt = 0, 0

    # for extraction
    system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)."

    input_file_name = "".join(args.input_file.split('.')[:-1]) # strip off file suffix .jsonl
    input_path = os.path.join(args.data_dir, args.input_file)
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    print('#################### extract claims ####################')

    prompt_template = prompt_formatter.get_prompt_template(qa_input=True)
    for dict_item in tqdm(data): # per question
        response = dict_item["response"]
        question = dict_item["question"]
        # prompt_source = dict_item["prompt_source"]
        # model = dict_item["model"]
        if args.extraction_method == 'chunk':
            sentences, snippet_lst = prompt_formatter.chunk_prompt(question, response, args.stride)
        elif args.extraction_method == 'sliding_window':
            sentences, snippet_lst = prompt_formatter.sliding_window_prompt(question, response)
        else:
            print("Please specify --extraction_method.")
            break

        for i, snippet in enumerate(snippet_lst): # per sentence/chunk
            prompt_text = prompt_template.format(snippet=snippet, sentence=sentences[i])
            prompt_tokens, response_tokens, total_tokens = estimator.estimate_tokens(system_message, prompt_text)
            total_prompt_tok_cnt += prompt_tokens
            total_resp_tok_cnt += response_tokens

    print(f"Prompt Tokens: {total_prompt_tok_cnt}")
    print(f"Estimated Response Tokens: {total_resp_tok_cnt}")
    cost = (5*total_prompt_tok_cnt + 15*total_resp_tok_cnt) / 1_000_000
    print(f"Total cost for gpt-4o (Estimate) in extraction: {cost}")

    print('\n#################### verify claims ####################')
    model_output_dir = os.path.join(args.output_dir, input_file_name, f"{args.extraction_method}_m=3_gpt-4o_gpt-4o")
    evidence_output_file = "evidence.jsonl"
    evidence_output_path = os.path.join(model_output_dir, evidence_output_file)
    
    if os.path.exists(evidence_output_path):
        # open evidence.jsonl
        with open(evidence_output_path, "r") as f:
            searched_evidence_dict = [json.loads(x) for x in f.readlines() if x.strip()]
        # get prompt_initial_temp and tail temp
        verify_system_message = "You are a helpful assistant who can judge whether a claim is supported or contradicted by the search results, or whether there is no enough information to make a judgement." # TODO: label = 3; add label = 2.
        prompt_initial_temp = prompt_formatter.get_verify_prompt_template()
        your_task = "Your task:\n\nClaim: {claim}\n\n{search_results}\n\nYour decision:"

        # fill question per dict_item
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """
        for dict_item in tqdm(searched_evidence_dict): # per question
            model_name = dict_item['model']
            domain = dict_item['prompt_source']
            claim_snippets_dict = dict_item["claim_search_results"]

            # skip if abstained
            if dict_item['abstained']:
                f.write(json.dumps(dict_item) + "\n")
                print("Skipping abstained")
                continue

            for claim, search_snippet_lst in claim_snippets_dict.items(): # per claim
                verify_prompt_tokens, verify_response_tokens = 0, 0
                search_res_str = ""
                search_cnt = 1
                for search_dict in search_snippet_lst[:search_res_num]: # per search_res_num (=10) result; add one by one
                    search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                    search_cnt += 1
                prompt_tail = your_task.format(
                            claim=claim,
                            search_results=search_res_str.strip(),) # fill the search results to the prompt
                verify_prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
                verify_prompt_tokens, verify_response_tokens, verify_total_tokens = estimator.estimate_tokens(verify_system_message, verify_prompt)
                total_verify_prompt_tok_cnt += verify_prompt_tokens
                total_verify_resp_tok_cnt += verify_response_tokens
        print(f"Prompt Tokens: {total_verify_prompt_tok_cnt}")
        print(f"Estimated Response Tokens: {total_verify_resp_tok_cnt}")
        verify_cost = (5*total_verify_prompt_tok_cnt + 15*total_verify_resp_tok_cnt) / 1_000_000
        print(f"Total cost for gpt-4o (Estimate) in extraction: {verify_cost}")
    else:
        print("No file named evidence.jsonl.")

    print('\n#################### Adding processes in total ####################')
    print(f"Prompt Tokens: {total_prompt_tok_cnt+total_verify_prompt_tok_cnt}")
    print(f"Estimated Response Tokens: {total_resp_tok_cnt+total_verify_resp_tok_cnt}")
    print(f"Total cost for gpt-4o (Estimate) in extraction: {cost+verify_cost}")