"""
This is the main script to evaluate factual score about the model responses.
It is comprised of three stages: (1) claim extraction, (2) evidence search, and (3) claim verification.
The (1) and (3) stage calls an LLM, and (2) uses a search API and SERP scraping API.
"""
# TODO: optimize the working flown for structure; add async for efficiency

import os
import json
import argparse
from collections import defaultdict
import spacy
from tqdm import tqdm
import time

from halluscore import utils
from .claim_extractor import ClaimExtractor
from .web_search_API import WebSearchAPI
from .claim_verifier import ClaimVerifier

class HalluScorer(object):
    def __init__(self,
                 model_name_extraction='gpt-4-0125-preview',
                 model_name_verification='gpt-4o',
                 use_external_extraction_model=False,
                 use_external_verification_model=False,
                 use_base_extraction_model=False,
                 use_base_verification_model=False,
                 data_dir='./data',
                 cache_dir='./data/cache',
                 output_dir='./data',
                 label_n=3,
                 pre_veri_label_m=3,
                 extraction_method='chunk',
                 stride=0,
                 verify_res_num=5,
                 search_res_num=5,
                 do_not_pre_verify=False,
                 logprob_threshold=float('-inf'),
                 ignore_cache=False
                 ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.system_message_extraction = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text and evaluate the factual correctness of each claim. Each atomic claim should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.claim_extractor = ClaimExtractor(model_name_extraction, pre_veri_label_m,
                                              extraction_method = extraction_method,         
                                              stride = stride,
                                              cache_dir=self.cache_dir,
                                              use_external_model=use_external_extraction_model, 
                                              use_base_model=use_base_extraction_model,
                                              do_not_pre_verify = do_not_pre_verify,
                                              logprob_threshold=logprob_threshold,
                                              )

        self.fetch_search = WebSearchAPI()

        demon_dir = os.path.join(self.data_dir, 'demos')
        self.model_name_verification = model_name_verification
        self.claim_verifier = ClaimVerifier(model_name=model_name_verification, label_n=label_n,
                                            cache_dir=self.cache_dir, demon_dir=demon_dir,
                                            use_external_model=use_external_verification_model, 
                                            use_base_model=use_base_verification_model)
        self.label_n = label_n
        self.pre_veri_label_m = pre_veri_label_m
        self.search_res_num = search_res_num
        self.verify_res_num = verify_res_num
        self.extraction_method = extraction_method
        self.do_not_pre_verify = do_not_pre_verify
        self.logprob_threshold = logprob_threshold

    def get_halluscore(self, data, input_file_name, model_name_extraction, model_name_verification, ignore_cache):
        shorter_model_name_extraction = model_name_extraction.split('/')[-1]
        shorter_model_name_verification = model_name_verification.split('/')[-1]
        model_output_dir = os.path.join(self.output_dir, input_file_name, f"{self.extraction_method}_m={self.pre_veri_label_m}_{shorter_model_name_extraction}_{shorter_model_name_verification}")
        
        # add suffix on the output folder
        if self.do_not_pre_verify:
            model_output_dir = f"{model_output_dir}_noPreverify"
        if self.logprob_threshold != float('-inf'):
            # Format -0.01 as '-0p01' to make it filesystem-friendly
            logprob_str = str(abs(self.logprob_threshold)).replace('.', 'p')
            if self.logprob_threshold < 0:
                logprob_str = f"neg{logprob_str}"
            model_output_dir = f"{model_output_dir}_logprob={logprob_str}"
        os.makedirs(model_output_dir, exist_ok=True)

        # Record the start time
        start_time = time.time()

        ######################################## extract and pre-verify claims ########################################
        claims_output_file = "claims.jsonl"
        claims_output_path = os.path.join(model_output_dir, claims_output_file)
        abstain_count = 0

        # Load saved items in the existing output file
        extracted_data = set()
        if os.path.exists(claims_output_path) and not ignore_cache:
            with open(claims_output_path, "r") as f:
                extracted_claims = [json.loads(line) for line in f]
            for item in extracted_claims:
                if 'predefined_stats_lst' in item and len(item['predefined_stats_lst']) == len(item['claim_to_verify_list']):
                    key = (item['response'], item['prompt_source'], item['model'])
                    extracted_data.add(key)
                    print(f"Loaded existing claim from {claims_output_path}")
                else:
                    extracted_claims = []
                    continue
        else:
            extracted_claims = []

        # Extract claims
        with open(claims_output_path, "w") as f:
            for dict_item in tqdm(data): # per question
                response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response, prompt_source, model)

                # skip the item if already extracted
                if key in extracted_data:
                    print(f"Skipping the item that already has extracted claims!")
                    continue

                # skip abstained responses
                if utils.is_abstain_response(response):
                    question = dict_item["question"] if "question" in dict_item else ''
                    claim_dict = {"question": question.strip(),
                                   "response": response.strip(),
                                   "abstained": True,
                                   "prompt_source": prompt_source,
                                   "model": model, }
                    f.write(json.dumps(claim_dict) + "\n")
                    extracted_data.add(key)
                    abstain_count += 1
                    continue

                # if not skipping, run claim_extractor
                question = dict_item["question"] if "question" in dict_item and dict_item["question"] else ''
                (
                    predefined_stats_lst, 
                    claim_to_verify_list, 
                    all_claims_to_verify, 
                    all_presupport_lst, 
                    all_preunsupport_lst, 
                    prompt_tok_cnt, 
                    response_tok_cnt
                ) = self.claim_extractor.run(question, response)

                # write output
                claim_dict = {"question": question.strip(),
                               "prompt_source": prompt_source,
                               "response": response.strip(),
                               "prompt_tok_cnt": prompt_tok_cnt,
                               "response_tok_cnt": response_tok_cnt,
                               "model": model,
                               "abstained": False,
                               "predefined_stats_lst": predefined_stats_lst, # NOTE: did not filter out the repeated identical claims
                               "pre_supported_claims_lst": all_presupport_lst,
                               "pre_unsupported_claims_lst": all_preunsupport_lst,
                               "claim_to_verify_list": claim_to_verify_list, # NOTE: did not filter out the repeated identical claims
                               "all_claims_to_verify": all_claims_to_verify # NOTE: filtered out the repeated identical claims!!!
                               }
                f.write(json.dumps(claim_dict) + "\n")
                f.flush()  # Ensure the item is written to the file immediately
                extracted_data.add(key) # update extracted_data to track this processed item
                extracted_claims.append(claim_dict)
        print(f"Claim extraction is done! saved to {claims_output_path}")


        ######################################## search evidence ########################################
        evidence_output_file = "retrieved_evidence.jsonl"
        evidence_output_path = os.path.join(model_output_dir, evidence_output_file)
        searched_evidences = []

        # load already searched items
        searched_data = set()
        if os.path.exists(evidence_output_path) and not ignore_cache:
            with open(evidence_output_path, "r") as f:
                existing_searched_evidences = [json.loads(line) for line in f]
            for dict_item in existing_searched_evidences:
                # Create a tuple key to identify already searched items
                if dict_item.get("retrieved_search_results") is not None:
                    response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                    key = (response, prompt_source, model)
                    searched_data.add(key)
            print(f"Loaded {len(searched_data)} existing searched items")
        

        with open(evidence_output_path, "w") as f:
            for dict_item in tqdm(extracted_claims): # per question response
                response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response, prompt_source, model)

                # Skip if this item has already been searched
                if key in searched_data:
                    print(f"Skipping already searched item.")
                    continue

                # Skip abstained responses (write, append, track)
                if dict_item['abstained']:
                    dict_item["claim_search_results"], dict_item["retrieved_search_results"] = {}, {}
                    f.write(json.dumps(dict_item) + "\n")
                    searched_evidences.append(dict_item)
                    searched_data.add(key)
                    continue
                
                claim_lst = dict_item["all_claims_to_verify"]
                # skip if no verifiable claim to search for evidence, for the entire response (make up, write, append, track)
                if claim_lst == ["No verifiable claim."] or claim_lst == []:
                    dict_item["claim_search_results"], dict_item["retrieved_search_results"] = {}, {}
                    f.write(json.dumps(dict_item) + "\n")
                    searched_evidences.append(dict_item)
                    searched_data.add(key)
                    continue
                
                # search evidence
                search_content_dict = self.fetch_search.get_content(claim_lst, self.search_res_num)
                dict_item["claim_search_results"] = search_content_dict

                # retrieve evidence
                retrieved_evidence_dict = {}
                for claim, evidence_list in search_content_dict.items():
                    retrieved_evidence_dict[claim] = utils.retrieve_relevant_passages(
                                                                                    claim, 
                                                                                    evidence_list, 
                                                                                    target_chunk_size=200, 
                                                                                    overlap=50, 
                                                                                    n=3
                                                                                    )
                dict_item["retrieved_search_results"] = retrieved_evidence_dict

                # (write, append, track)
                searched_evidences.append(dict_item)
                f.write(json.dumps(dict_item) + "\n")
                searched_data.add(key)
                f.flush()  # Ensure the item is written to the file immediately

        print(f"Knowledge searching is done! saved to {evidence_output_path}")

        ######################################## verify claims ##########################################
        verification_output_file = f'verification_label_n={self.label_n}.jsonl'
        veriscore_file = f'veriscore_label_n={self.label_n}.csv'
        verification_output_path = os.path.join(model_output_dir, verification_output_file)
        veriscore_path = os.path.join(model_output_dir, veriscore_file)

        model_domain_stats_dict , model_domain_response_stats_dict = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))  
        total_prompt_tok_cnt = 0
        total_resp_tok_cnt = 0

        # load already verified items
        verified_data = set()
        if os.path.exists(verification_output_path) and not ignore_cache:
            with open(verification_output_path, "r") as f:
                existing_verified_items = [json.loads(line) for line in f]
            for dict_item in existing_verified_items:
                # Create a tuple key to identify already verified items
                if dict_item.get("retrieved_search_results") is not None:
                    response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                    key = (response, prompt_source, model)
                    verified_data.add(key)
            print(f"Loaded {len(verified_data)} existing verified items")
        # TODO: aggregate the record loading and handling in three sections into a modular function!

        # verify
        with open(verification_output_path, "w") as f:
            for dict_item in tqdm(searched_evidences): # for each question
                # skip if already verified (write, append, track)
                response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response, prompt_source, model)
                if key in verified_data:
                    print("Skipping already verified item.")
                    f.write(json.dumps(dict_item) + "\n")  # Still write the item to maintain consistency
                    verified
                    continue

                # skip if abstained or no claims to verify (write, append, track)
                if dict_item['abstained']:
                    f.write(json.dumps(dict_item) + "\n")

                    print("Skipping abstained/no claims to check")
                    continue

                model_name = dict_item['model']
                domain = dict_item['prompt_source']
                retrieved_search_results = dict_item.get("retrieved_search_results") # list({'retrieved_text':[str or empty]})

                # call verifier
                claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = self.claim_verifier.verifying_claim(
                    retrieved_search_results, verify_res_num=args.verify_res_num)
                dict_item["claim_verification_result"] = claim_verify_res_dict

                # accumulate token_cnt for calculating Total cost
                total_prompt_tok_cnt += prompt_tok_cnt
                total_resp_tok_cnt += response_tok_cnt

                # accumulate stats for calculating final avg score
                stats = [0, 0, 0]
                '''
                stats:
                [number of all supported claims (i.e., support score), 
                number of all claims, 
                number of sentences]
                '''
                dict_item['response_stats'] = defaultdict(lambda: defaultdict(int)) # initiate a nested dict

                ### 2. sentence count
                stats[2] = len(dict_item['claim_to_verify_list'])
                dict_item['response_stats']['sentences'] = stats[2]

                ### 1. claim count (verified claims)
                stats[1] = len(dict_item['all_claims_to_verify'])
                dict_item['response_stats']['verified_claims'] = stats[1]
                
                ### 0. Support count
                # (verified supported claims)
                if not dict_item['claim_search_results']:
                    stats[0] = 0
                else:
                    for claim_veri_res in dict_item.get('claim_verification_result', []):
                        if claim_veri_res['verification_result'] == "supported":
                            stats[0] += 1
                print(f"\nStats before adding pre-support score: {stats} ([verified supported claims, verified claims, sentences])")
                dict_item['response_stats']['verified_supported_claims'] = stats[0]

                # (pre-verified supported claims)
                pre_supported = sum(predefined_stats[1] for predefined_stats in dict_item["predefined_stats_lst"])
                claims_to_check = sum(predefined_stats[0] for predefined_stats in dict_item["predefined_stats_lst"])
                pre_unsupported = sum(predefined_stats[2] for predefined_stats in dict_item["predefined_stats_lst"])
                print(f"Predefined stats total for the question: {pre_supported}, {claims_to_check}, {pre_unsupported} (sum of [pre-support_claims, unsure_claims, pre-unsupport_claims])")
                stats[0] += pre_supported
                dict_item['response_stats']['pre_supported_claims'] = pre_supported
                dict_item['response_stats']['pre_verified_claims'] = pre_supported + pre_unsupported

                ### 1. claim count (all claims)
                # Check how many identical claims are filtered:
                if len(dict_item['all_claims_to_verify']) != claims_to_check:
                    print(f"{claims_to_check - len(dict_item['all_claims_to_verify'])} identical claim(s) are filtered out in all_claims_to_verify.")
                stats[1] = pre_supported + len(dict_item['all_claims_to_verify']) + pre_unsupported
                print(f"Stats after adding pre-support score: {stats} ([supported claims, all claims, sentences])\n")
                dict_item['response_stats']['all_claims'] = stats[1]

                dict_item['response_stats']['P'] = (dict_item['response_stats']['pre_supported_claims'] + dict_item['response_stats']['verified_supported_claims']) / dict_item['response_stats']['all_claims'] if dict_item['response_stats']['all_claims'] != 0 else None
                
                # save and append the question stats (write, append, track)
                f.write(json.dumps(dict_item) + "\n")
                f.flush()
                model_domain_stats_dict[domain][model_name].append(stats)
                model_domain_response_stats_dict[domain][model_name].append(dict_item['response_stats'])
                verified_data.add(key)
            print(f"Claim verification is saved to {verification_output_path}")
        
        ### Calculate F1@k ###
        print(f"Calculating and saving veriscores to {veriscore_path}")
        end_time = time.time()
        total_time = end_time - start_time

        utils.get_halluscore(model_domain_stats_dict, model_domain_response_stats_dict,
                            veriscore_path, 
                            model_name_extraction, model_name_verification, 
                            total_prompt_tok_cnt, total_resp_tok_cnt, total_time,
                            abstain_count, len(data))
        print(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}") # 10 / 1e6 means $0.00001 per token
        print(f"Time used: {total_time}.")
        print(f"Abstained response rate: {abstain_count}/{len(data)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--cache_dir", type=str, default='./data/cache')
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--model_name_verification", type=str, default="gpt-4o")
    parser.add_argument("--label_n", type=int, default=3, choices=[2, 3])
    parser.add_argument("--pre_veri_label_m", type=int, default=3, choices=[3, 5])
    parser.add_argument("--extraction_method", type=str, required=True, default='chunk', choices=['chunk', 'sliding_window'], help='sliding_window: (context1 = 0-3 sentence) <SOS>Sentence to be focused on<EOS> (context2 = 0-1 sentence); chunk:') # TODO: complete description
    parser.add_argument("--stride", type=int, default=0, help='You can specify a fixed stride in chunking; 0 means feeding the whole response for extraction; -1 means dynamic stride based on response length.', required=False)
    parser.add_argument("--search_res_num", type=int, default=5, help='the number of evidence results to search for and save.')
    parser.add_argument("--verify_res_num", type=int, default=5, help='the number of evidence results used for verification.')
    parser.add_argument("--use_external_extraction_model", action='store_true')
    parser.add_argument("--use_external_verification_model", action='store_true')
    parser.add_argument("--use_base_extraction_model", action='store_true')
    parser.add_argument("--use_base_verification_model", action='store_true')
    parser.add_argument("--do_not_pre_verify", action='store_true')
    parser.add_argument("--logprob_threshold", type=float, default=float('-inf'))
    parser.add_argument("--ignore_cache", action='store_true')
    args = parser.parse_args()

    vs = HalluScorer(model_name_extraction=args.model_name_extraction,
                    model_name_verification=args.model_name_verification,
                    use_external_extraction_model=args.use_external_extraction_model,
                    use_external_verification_model=args.use_external_verification_model,
                    use_base_extraction_model=args.use_base_extraction_model,
                    use_base_verification_model=args.use_base_verification_model,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    cache_dir=args.cache_dir,
                    label_n=args.label_n,
                    pre_veri_label_m = args.pre_veri_label_m,
                    extraction_method = args.extraction_method,
                    stride = args.stride,
                    search_res_num=args.search_res_num,
                    verify_res_num=args.verify_res_num,
                    do_not_pre_verify=args.do_not_pre_verify,
                    logprob_threshold=args.logprob_threshold,
                    ignore_cache=args.ignore_cache,
                    )

    input_file_name = "".join(args.input_file.split('.')[:-1]) # strip off file suffix .jsonl
    input_path = os.path.join(args.data_dir, args.input_file)
    # read jsonl in lists of json
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    vs.get_halluscore(data, input_file_name, args.model_name_extraction, args.model_name_verification, args.ignore_cache)
