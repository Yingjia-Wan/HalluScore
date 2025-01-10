import os
import regex
import pdb
import json
import spacy
from tqdm import tqdm
from .response_API import GetResponse
from .utils import *
import re


class ClaimExtractor():
    def __init__(self, model_name, label_m, extraction_method, stride=None, cache_dir="./data/cache/", use_external_model=False, use_base_model=False, do_not_pre_verify=False, logprob_threshold=float('-inf')):
        self.model = None
        self.label_m = label_m
        self.extraction_method = extraction_method
        self.stride = stride
        self.do_not_pre_verify = do_not_pre_verify
        self.logprob_threshold = logprob_threshold

        if os.path.isdir(model_name) or use_external_model:
            from unsloth import FastLanguageModel
            # doc: https://github.com/unslothai/unsloth/blob/e32fc240884435527660bb79a5664a94e27a7576/unsloth/models/loader.py#L70
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1000,
                dtype=None,
                load_in_4bit=True,
                device_map = "sequential" # Unsloth does not support multi GPU settings currently.
            )
            FastLanguageModel.for_inference(self.model)
            self.model = self.model.to("cuda")
            self.alpaca_prompt = open("./prompt/extraction/extraction_alpaca_template.txt", "r").read()

        elif use_base_model:
            # Load base LLM like meta-llama/Meta-Llama-3-8B-Instruct
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            device_map = get_device_map()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.alpaca_prompt = open("./prompt/extraction/extraction_alpaca_template.txt", "r").read()

        else:
            # Use gpt or claude
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0)
            self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def run(self, question, response):
        if question == '':
            return self.non_qa_scanner_extractor(response)
        else:
            return self.qa_scanner_extractor(question, response)

    def non_non_qa_scanner_extractor(self, response, cost_estimate_only=False):
        # TODO: sync non_qa with qa improvements
        """
        Given a model output
        - split by \n into paragraphs
        - split the paragraphs into sentences using spaCy
        - go para by para, always add the first sent of the para into context1
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        # split response into paras & clean out empty strings
        paragraph_lst = [x.strip() for x in response.split("\n") if x.strip() != ""]

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        for para in paragraph_lst:
            # split the text into sentences using spaCy
            sentences = self.get_sentence(para)
            for i, sentence in enumerate(sentences):
                if self.model:
                    input = response.strip()
                    snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
                else:
                    lead_sent = sentences[0]  # 1st sentence of the para
                    context1 = " ".join(sentences[max(0, i - 3):i])
                    sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                    context2 = " ".join(sentences[i + 1:i + 2])

                    # if the para is not long
                    if len(sentences) <= 5:
                        snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                    # if the para is long, add lead sentence to context1
                    else:
                        snippet = f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()

                # call fact_extractor on each snippet
                facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(),
                                                                              qa_input=False,
                                                                              cost_estimate_only=cost_estimate_only)

                # update token counts
                prompt_tok_cnt += prompt_tok_num
                response_tok_cnt += response_tok_num

                if facts == None:
                    continue

                # deduplication
                for fact in facts:
                    if fact.strip() == "":
                        continue
                    # cases where GPT returns its justification
                    elif fact.startswith("Note:"):
                        continue
                    elif fact not in all_facts_lst:
                        all_facts_lst.append(fact)

        print(f"Returning facts and token counts for the whole response ...")
        if all_facts_lst == None:
            return None, prompt_tok_cnt, response_tok_cnt
        else:
            return all_facts_lst, prompt_tok_cnt, response_tok_cnt

    def qa_scanner_extractor(self, question, response, cost_estimate_only=False):
        """
        Given a model output to a question
        - split the response into sentences using spaCy
        - snippet = question (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """  
        if self.extraction_method == 'sliding_window':
            predefined_triplet_lst, fact_lst_lst, all_facts_lst, all_presupport_lst, all_preunsupport_lst, prompt_tok_cnt, response_tok_cnt = self.extract_using_sliding_window(response, question, cost_estimate_only)
        elif self.extraction_method == 'chunk':
            predefined_triplet_lst, fact_lst_lst, all_facts_lst, all_presupport_lst, all_preunsupport_lst, prompt_tok_cnt, response_tok_cnt = self.extract_using_chunk(response, question, cost_estimate_only)

        # Sanity Check: the total number of unsured claims == that of the all claims to verify 
        # NOTE: all fact_lst_lst/predefined_triplet/all_facts_lst has filtered out the repeated identical claims.
        num_unsured_claims = sum(x[0] for x in predefined_triplet_lst if x)
        claims_to_check = sum(len(fact_lst) for fact_lst in fact_lst_lst if fact_lst)
        # without filtering out the repeated identical claims:
        assert num_unsured_claims == claims_to_check, \
            f"Mismatch in claim_to_check count: num_unsured_claims {num_unsured_claims} != claims_to_check {claims_to_check}"

        return predefined_triplet_lst, fact_lst_lst, all_facts_lst, all_presupport_lst, all_preunsupport_lst, prompt_tok_cnt, response_tok_cnt


    def extract_using_sliding_window(self, response, question, cost_estimate_only):
        sentences = self.get_sentence(response)
        # new return values
        prompt_tok_cnt, response_tok_cnt = 0, 0
        snippet_lst = []
        fact_lst_lst = [] # a list of fact_list per sentence to be verified
        all_facts_lst = [] # a list of all (unsure) facts to be verified
        all_presupport_lst = [] # a list of all presupport facts
        all_preunsupport_lst = [] # a list of all preunsupport facts
        predefined_triplet_lst = []

        for i, sentence in enumerate(sentences):
            if self.model:
                input = f"Questions:\n{question.strip()}\nResponse:\n{response.strip()}"
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])
                snippet = f"Question: {question.strip()}\nResponse: {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            # new return value
            snippet_lst.append(snippet)

            # call fact_extractor (model) on each snippet text (sliding window prompt for each sentence)
            unsure_claims, support_claims, unsupport_claims, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(), qa_input=True,
                                                                          cost_estimate_only=cost_estimate_only) # claims per sentence
            
            # remove empty lines
            unsure_claims, support_claims, unsupport_claims = [claim for claim in unsure_claims if claim and not claim.startswith("Note:")], [claim for claim in support_claims if claim and not claim.startswith("Note:")], [claim for claim in unsupport_claims if claim and not claim.startswith("Note:")]

            # update token counts
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            # reinitialize fact_lst for each sentence
            fact_lst = []
            presupport_count, preunsupport_count, unsure_count = 0, 0, 0
            for fact in support_claims:
                if fact not in all_presupport_lst: # check duplication
                    all_presupport_lst.append(fact.strip())
                    presupport_count += 1

            for fact in unsupport_claims:
                if fact not in all_preunsupport_lst:
                    all_preunsupport_lst.append(fact.strip())
                    preunsupport_count += 1

            for fact in unsure_claims:
                if fact not in all_facts_lst: # check duplication in unsure_claims
                    all_facts_lst.append(fact.strip())
                    unsure_count += 1
                    fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)

            # calculate predefined_triplet
            predefined_triplet = [unsure_count, presupport_count, preunsupport_count]
            predefined_triplet_lst.append(predefined_triplet)
        # print(f"Returning predefined_triplet_lst, prompts, facts and token counts for the question response ...")
        return predefined_triplet_lst, fact_lst_lst, all_facts_lst, all_presupport_lst, all_preunsupport_lst, prompt_tok_cnt, response_tok_cnt

    def extract_using_chunk(self, response, question, cost_estimate_only):
        chunks = self.get_chunk(response, self.stride)

        # new return values
        prompt_tok_cnt, response_tok_cnt = 0, 0
        snippet_lst = []
        fact_lst_lst = [] # a list of fact_list per sentence to be verified
        all_facts_lst = [] # a list of all (unsure) facts to be verified
        all_presupport_lst = [] # a list of all presupport facts
        all_preunsupport_lst = [] # a list of all preunsupport facts
        predefined_triplet_lst = []

        for i, chunk in enumerate(chunks):
            snippet = f"Question: {question.strip()}\nResponse: <SOS>{chunk.strip()}<EOS>".strip()
            # new return value
            snippet_lst.append(snippet)

            # call fact_extractor (model) on each snippet text (sliding window prompt for each sentence)
            unsure_claims, support_claims, unsupport_claims, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, "", qa_input=True,
                                                                          cost_estimate_only=cost_estimate_only) # claims per sentence

            # update token counts
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num
            
            # remove empty lines and explanations
            unsure_claims, support_claims, unsupport_claims = [claim for claim in unsure_claims if claim and not claim.startswith("Note:")], [claim for claim in support_claims if claim and not claim.startswith("Note:")], [claim for claim in unsupport_claims if claim and not claim.startswith("Note:")]

            # reinitialize fact_lst for each chunk
            fact_lst = []
            presupport_count, preunsupport_count, unsure_count = 0, 0, 0
            for fact in support_claims:
                if fact not in all_presupport_lst: # check duplication
                    all_presupport_lst.append(fact.strip())
                    presupport_count += 1

            for fact in unsupport_claims:
                if fact not in all_preunsupport_lst:
                    all_preunsupport_lst.append(fact.strip())
                    preunsupport_count += 1

            for fact in unsure_claims:
                if fact not in all_facts_lst: # check duplication in unsure_claims
                    all_facts_lst.append(fact.strip())
                    unsure_count += 1
                    fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)

            # calculate predefined_triplet
            predefined_triplet = [unsure_count, presupport_count, preunsupport_count]
            predefined_triplet_lst.append(predefined_triplet)
        # print(f"Returning predefined_triplet_lst, prompts, facts and token counts for the question response ...")
        return predefined_triplet_lst, fact_lst_lst, all_facts_lst, all_presupport_lst, all_preunsupport_lst, prompt_tok_cnt, response_tok_cnt

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]
    
    def get_chunk(self, text, stride):
        '''
        stride means the sentence count of a chunk. 
        So you have the options to set:
            a fixed stride (stride = 0 means whole response no stride; other positive intergers mean the actual stride) 
            or a dynamic one based on response length (stride = -1).
        '''
        if stride == 0:
            return [text] # feed the extractor the whole response 
            #TODO: handle the edge case if response exceeds the context length
        # TODO: elif stride == -1:
        chunks = []
        sentences = self.get_sentence(text)
        for i in range(0, len(sentences), stride):
            chunk = sentences[i:i + stride] # TODO: overlapping context
            chunks.append(' '.join(chunk))
        return chunks

    def get_prompt_template(self, qa_input):
        if self.do_not_pre_verify:
            if qa_input:
                prompt_template = open(f"./prompt/extraction/{self.extraction_method}_m={self.label_m}_only_extraction_qa_template.txt", "r").read()
            else:
                # prompt_template = open(f"./prompt/extraction/{self.extraction_method}_m={self.label_m}_only_extraction_non_qa_template.txt", "r").read()
                print("Prompt template for non-qa not-pre-verify is not in the folder.")
        else:
            if qa_input:
                prompt_template = open(f"./prompt/extraction/{self.extraction_method}_m={self.label_m}_extraction_qa_template.txt", "r").read()
            else:
                prompt_template = open(f"./prompt/extraction/{self.extraction_method}_m={self.label_m}_extraction_non_qa_template.txt", "r").read()
        return prompt_template

    def fact_extractor(self, snippet, sentence, qa_input=False, cost_estimate_only=False):
        """
        if sliding_window:
            snippet = (context1) <SOS>sentence<EOS> (context2)
            sentence = the sentence to be focused on
        if chunk:
            snippet = (context1) <SOS>chunk<EOS> (context2)
            sentence = "" (unused because irrelevant)
        """
        ### Extract verifiable claims via base/finetuned LLMs
        if self.model:
            formatted_input = self.alpaca_prompt.format(snippet, "")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=1000, use_cache=True)
            output_str = ' '.join(self.tokenizer.batch_decode(outputs))
            
            clean_output = output_str.split("### Response:")[-1].strip().replace("</s>", "")

            if not clean_output or "no verifiable claim" in clean_output.lower():
                return None, [], [], 0, 0

            claims = [x.strip() for x in clean_output.split("\n")]
            
            unsure_claims = []
            support_claims = []
            unsupport_claims = []

            for claim in claims:
                if claim.endswith("###UNSURE###"):
                    unsure_claims.append(claim.replace("###UNSURE###", "").strip())
                elif claim.endswith("###TRUE###"):
                    support_claims.append(claim.replace("###TRUE###", "").strip())
                elif claim.endswith("###FALSE###"):
                    unsupport_claims.append(claim.replace("###FALSE###", "").strip())
                else:
                    unsure_claims.append(claim.strip())
            return unsure_claims, support_claims, unsupport_claims, 0, 0

        else:
            ### Prompting base approach via API call
            prompt_template = self.get_prompt_template(qa_input)  # qa_prompt_temp if qa_input else non_qa_prompt_temp

            if self.extraction_method == 'sliding_window':
                prompt_text = prompt_template.format(snippet=snippet, sentence=sentence)
            elif self.extraction_method == 'chunk':
                prompt_text = prompt_template.format(snippet=snippet) # no {sentence}

            # get response
            response, logprobs, prompt_tok_cnt, response_tok_cnt = self.get_model_response.get_response(self.system_message,
                                                                                            prompt_text,
                                                                                            cost_estimate_only,
                                                                                            self.logprob_threshold)
            # print(response)
            if not response or "no verifiable claim" in response.lower():
                return [], [], [], prompt_tok_cnt, response_tok_cnt
            else:
                # remove noisy formats in response to filter out claims
                claims = clean_claim_format(response)

                unsure_claims = []
                support_claims = []
                unsupport_claims = []

                for claim in claims:

                    if claim.endswith("###UNSURE###") or claim.endswith("###LIKELY TRUE###") or claim.endswith("###LIKELY FALSE###"):
                        cleaned = clean_claim_label(claim, ["###UNSURE###", "###LIKELY TRUE###", "###LIKELY FALSE###"])
                        unsure_claims.append(cleaned)

                    elif claim.endswith("###TRUE###"):
                        # check certainty from logprobs
                        label_logprob = check_token_logprobs(claim, logprobs)
                        if label_logprob >= self.logprob_threshold:
                            support_claims.append(claim.replace("###TRUE###", "").strip())
                        else:
                            unsure_claims.append(claim.replace("###TRUE###", "").strip())

                    elif claim.endswith("###FALSE###"):
                        # check certainty from logprobs
                        label_logprob = check_token_logprobs(claim, logprobs)
                        if label_logprob >= self.logprob_threshold:
                            unsupport_claims.append(claim.replace("###FALSE###", "").strip())
                        else:
                            unsure_claims.append(claim.replace("###FALSE###", "").strip())
                            
                    else: # if no label is attached or other labels like ###LIKELY TRUE### is attached
                        cleaned_claim = re.sub(r'###[^#]+###', '', claim).strip()
                        unsure_claims.append(cleaned_claim)

                return unsure_claims, support_claims, unsupport_claims, prompt_tok_cnt, response_tok_cnt
