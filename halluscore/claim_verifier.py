import os
import pdb
import json

from .response_API import GetResponse
from .utils import get_device_map


class ClaimVerifier():
    def __init__(self, model_name, label_n=2, cache_dir="./data/cache/", demon_dir="data/demos/",
                 use_external_model=False, use_base_model=False):
        self.model = None
        self.model_name = model_name
        self.label_n = label_n
        if os.path.isdir(model_name) or use_external_model:
            # Load finetuned Unsloth model
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                device_map = "sequential" # Unsloth does not support multi GPU settings currently.
            )
            FastLanguageModel.for_inference(self.model)
            self.tokenizer.padding_side = "left"
            self.alpaca_prompt = open("./prompt/verification/verification_alpaca_template.txt", "r").read()
            self.instruction = open("./prompt/verification/verification_instruction_binary_no_demo.txt", "r").read()
        if use_base_model:
            # Load base LLM like meta-llama/Meta-Llama-3-8B-Instruct
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            device_map = get_device_map()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device_map # Unsloth does not support multi GPU settings currently.
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.alpaca_prompt = open("./prompt/verification/verification_alpaca_template.txt", "r").read()
            self.instruction = open("./prompt/verification/verification_instruction_binary_no_demo.txt", "r").read()
        else:
            # Use gpt or claude
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
            self.demon_path = os.path.join(demon_dir, 'few_shot_examples.jsonl')
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0)
            self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against reliable external world knowledge."
            self.prompt_initial_temp = self.get_initial_prompt_template()

    def get_instruction_template(self):
        prompt_temp = ''
        if self.label_n == 2 and "claude" in self.model_name:
            prompt_temp = open("./prompt/verification/verification_instruction_claude_binary.txt", "r").read()
        elif self.label_n == 3 and "claude" in self.model_name:
            prompt_temp = open("./prompt/verification/verification_instruction_claude_trinary.txt", "r").read()
        elif self.label_n == 2 and "claude" not in self.model_name:
            prompt_temp = open("./prompt/verification/jina_search/verification_instruction_binary.txt", "r").read()
        elif self.label_n == 3 and "claude" not in self.model_name:
            prompt_temp = open("./prompt/verification/jina_search/verification_instruction_trinary.txt", "r").read()
        return prompt_temp

    def get_initial_prompt_template(self):
        prompt_temp = self.get_instruction_template()
        with open(self.demon_path, "r") as f:
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

        prompt_few_shot = prompt_temp.format(*element_lst)

        if "claude" in self.model_name:
            self.your_task = "Your task:\n\n{search_results}\n\nClaim: {claim}\n\nTask: Given the search results above, is the claim supported or unsupported? Mark your decision with ### signs.\n\nYour decision:"
        else:
            self.your_task = "Your task:\n\nClaim: {claim}\n\n{search_results}\n\nYour decision:"

        """
        choose system message
        """
        if self.label_n == 2:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported by the search results or not."
        elif self.label_n == 3:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or contradicted by the search results, or whether there is no enough information to make a judgement."

        return prompt_few_shot

    def verifying_claim(self, search_results_dict, verify_res_num=5, cost_estimate_only=False):
        """
        search_results_dict = [{"title": title, "retrieved_text": list(str or empty) ...]
        list({'retrieved_text':[str or empty]})
        """

        prompt_tok_cnt, response_tok_cnt = 0, 0
        out_lst = []
        claim_verify_res_dict = {}
        for claim, claim_evidence_lst in search_results_dict.items():
            search_res_str = ""
            search_cnt = 1

            # Process up to `verify_res_num` search results for the current claim
            for search_dict in claim_evidence_lst[:verify_res_num]:
                retrieved_text = "\n".join(search_dict["retrieved_text"]).strip() if search_dict["retrieved_text"] != [] else ""
                search_res_str += (
                    f'Evidence {search_cnt}\n'
                    f'Title: {search_dict["title"].strip()}\n'
                    f'Content: {retrieved_text}\n\n'
                )
                search_cnt += 1

            if self.model:
                usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"
                formatted_input = self.alpaca_prompt.format(self.instruction, usr_input)

                inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
                output = self.model.generate(**inputs,
                                             max_new_tokens=500,
                                             use_cache=True,
                                             eos_token_id=[self.tokenizer.eos_token_id,
                                                           self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                                             pad_token_id=self.tokenizer.eos_token_id, )
                response = self.tokenizer.batch_decode(output)
                clean_output = ' '.join(response).split("<|end_header_id|>\n\n")[
                    -1].replace("<|eot_id|>", "").strip()
                
            else:
                prompt_tail = self.your_task.format(
                    claim=claim,
                    search_results=search_res_str.strip(),
                )
                prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
                response, logprobs, prompt_tok_num, response_tok_num = self.get_model_response.get_response(self.system_message,
                                                                                                  prompt,
                                                                                                  cost_estimate_only=cost_estimate_only,
                                                                                                  logprob_threshold=float('-inf'))
                                                                                                  #TODO: verification logprob threshold
                prompt_tok_cnt += prompt_tok_num
                response_tok_cnt += response_tok_num

                clean_output = response.replace("#", "").split(".")[0].lower() if response is not None else None
            claim_verify_res_dict = {"claim": claim,
                                     "search_results": search_res_str,
                                     "verification_result": clean_output}
            out_lst.append(claim_verify_res_dict)
        return out_lst, prompt_tok_cnt, response_tok_cnt
