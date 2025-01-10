'''
Example command:
python load_data.py \
--input_file_path 'factscore/ChatGPT.jsonl' \
--output_file_path 'factscore/ChatGPT_sample20.jsonl' \
--num_lines 20 \
--dataset_type 'factscore'

python load_data.py \
--input_file_path 'natyou/freshqa_10_06' \
--output_file_path 'freshQA/humans_sample20.jsonl' \
--num_lines 20 \
--dataset_type 'freshQA'

python load_data.py \
--input_file_path 'longfact/gpt35turbo_sample10.json' \
--output_file_path 'longfact/gpt35turbo_sample10.jsonl' \
--dataset_type 'longfact'

python load_data.py \
--input_file_path 'freshQA/gpt35turbo_sample20.json' \
--output_file_path 'freshQA/gpt35turbo_sample20.jsonl' \
--dataset_type 'freshQA'

python load_data.py \
--input_file_path 'felm/all.jsonl' \
--output_file_path './felm' \
--dataset_type 'felm' \
--num_lines 20

python load_data.py \
--input_file_path './ExpertQA/r2_compiled_anon.jsonl' \
--output_file_path './ExpertQA/all.jsonl' \
--dataset_type 'ExpertQA' \
--num_lines 20

python load_data.py \
--input_file_path './factcheck-GPT/factcheck-GPT-benchmark.jsonl' \
--output_file_path './factcheck-GPT/all.jsonl' \
--dataset_type 'factcheck-GPT'

Types of dataset to load:
'''

import json
from datasets import load_dataset
import argparse
import os

import json
import uuid

def check_uuid_in_jsonl(jsonl_path):
    """
    Check if all samples in the JSONL file contain the 'uuid' field.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        bool: True if all samples contain the 'uuid' field, False otherwise.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            sample = json.loads(line)
            
            # Check if 'uuid' field is present in the sample
            if 'uuid' not in sample:
                print(f"Error: Sample at line {line_number} is missing the 'uuid' field.")
                return False
    
    print("All samples contain the 'uuid' field.")
    return True

def add_uuid_to_jsonl_inplace(input_jsonl_path):
    """
    Add a UUID to each sample in the input JSONL file in place.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
    """
    # Read all lines into memory
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Modify each line by adding a UUID
    updated_lines = []
    for line in lines:
        input_data = json.loads(line)
        
        # Generate a UUID for the sample
        sample_uuid = str(uuid.uuid4())
        
        # Add the UUID to the sample
        input_data['uuid'] = sample_uuid
        
        # Append the updated sample to the list of updated lines
        updated_lines.append(json.dumps(input_data) + '\n')

    # Write the updated lines back to the same file
    with open(input_jsonl_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(updated_lines)


def load_felm(input_file_path, output_dir, num_lines):
    # Load the dataset from Hugging Face
    subsets = ['math', 'reasoning', 'science', 'wk', 'writing_rec']
    
    for subset in subsets:
        output_file_path = os.path.join(output_dir, f"{subset}.jsonl")
        dataset = load_dataset('hkust-nlp/felm', subset, split = 'test')
        
        if num_lines is None:
            num_lines = len(dataset)
        
        # Open the output file
        with open(output_file_path, 'w') as outfile:
            for index, data in enumerate(dataset):
                if index >= num_lines:
                    break
                
                labels = data.get("labels", [])
                # Calculate felm_gt_P as the percentage of true values in labels
                if len(labels) > 0:
                    felm_gt_P = sum(labels) / len(labels)
                else:
                    felm_gt_P = 0  # Handle case where labels might be empty

                converted_data = {
                    "question": data.get("prompt", ""),
                    "response": data.get("response", ""),
                    "model": "ChatGPT",
                    "prompt_source": "felm",
                    "felm_gt": labels,
                    "felm_gt_P": felm_gt_P,
                }
                outfile.write(json.dumps(converted_data) + '\n')


def convert_factscore_jsonl(input_file_path, output_file_path, num_lines):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        if num_lines == None:
            num_lines = len(infile)
        for index, line in enumerate(infile):
            if index >= num_lines:
                break
            data = json.loads(line)
            converted_data = {
                "question": data.get("input", ""),
                "response": data.get("output", ""),
                "model": "ChatGPT",
                "prompt_source": "factscore",
                "annotations": data.get("annotations", ""),
            }
            outfile.write(json.dumps(converted_data) + '\n')

def convert_freshqa_jsonl(input_file_path, output_file_path, num_lines=None):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        data = json.load(infile)

        if num_lines is None:
            num_lines = len(data)

        for index, item in enumerate(data):
            if index >= num_lines:
                break

            response = item.get("response", "")
            
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = response
            
            converted_data = {
                "question": item.get("question", ""),
                "response": response_text,
                "model": "gpt-3.5-turbo",
                "prompt_source": "freshQA",
                # "annotations": item.get("annotations", ""),
            }
            outfile.write(json.dumps(converted_data) + '\n')


def convert_longfact_jsonl(input_file_path, output_file_path, num_lines):
    with open(input_file_path, 'r') as input_file:
        data = json.load(input_file)
    per_prompt_data = data.get('per_prompt_data', [])
    if num_lines == None:
            num_lines = len(per_prompt_data)
    with open(output_file_path, 'w') as output_file:
        for i, prompt_dict in enumerate(per_prompt_data):
            if i >= num_lines:
                break
            question = prompt_dict.get('prompt', '')
            response = prompt_dict.get('side2_response', '')
            
            jsonl_entry = {
                "question": question,
                "response": response,
                "model": "gpt-3.5-turbo-0125",
                "prompt_source": "longfact",
            }
            
            output_file.write(json.dumps(jsonl_entry) + '\n')

def convert_expertqa_jsonl(input_jsonl_path, output_jsonl_path, num_lines=None):
    """
    Convert the input JSONL format to the desired output JSONL format.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
        output_jsonl_path (str): Path to the output JSONL file.
        num_lines (int, optional): Number of lines to convert. If None, convert all lines.
    """
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, open(output_jsonl_path, 'w', encoding='utf-8') as outfile:  
        line_count = 0
        skipped_lines = 0
        for line in infile:
            if num_lines is not None and line_count >= num_lines:
                break
            
            input_data = json.loads(line)
            
            # Extract the model name from the 'answers' key
            model_name = list(input_data['answers'].keys())[0]
            
            # Extract the response and annotations
            response_data = input_data['answers'][model_name]
            annotations = {
                'response_usefulness': response_data['usefulness'],
                'claims': [
                    {
                        'spacy_claim': claim.get('claim_string', None),
                        'human_revised_evidence': claim.get('revised_evidence', None),
                        'evidence_support': claim.get('support', None),
                        'reason_missing_support': claim.get('reason_missing_support', None),
                        'informativeness': claim.get('informativeness', None),
                        'worthiness': claim.get('worthiness', None),
                        'factual_correctness': claim.get('correctness', None),
                        'reliability': claim.get('reliability', None)
                    }
                    for claim in response_data.get('claims', [])
                ]
            }

            # Check and filter if any claim has None for factual_correctness or worthiness
            skip_line = False
            for claim in annotations['claims']:
                if claim['factual_correctness'] is None or claim['worthiness'] is None:
                    skip_line = True
                    break
            if skip_line:
                skipped_lines += 1
                continue

            # add gt_facteval_score: for each claim, 1 if {Factuality | label in {Definitely correct, Probably correct}} AND {Cite-worthiness | label == Yes}
            for claim in annotations['claims']:
                if claim['factual_correctness'].lower() in ['definitely correct','probably correct'] and claim['worthiness'].lower() == 'yes':
                    claim['facteval_score'] = 1
                else:
                    claim['facteval_score'] = 0
            gt_facteval_score = sum(claim['facteval_score'] for claim in annotations['claims'])
            
            # Construct the output dictionary
            output_data = {
                'uuid': input_data['uuid'],
                'field': input_data['metadata']['field'],
                'question': input_data['question'],
                'response': response_data['answer_string'],
                'model': model_name,
                'prompt_source': 'ExpertQA',
                'annotations': annotations,
                'gt_facteval': gt_facteval_score
            }
            
            # Write the output dictionary to the output JSONL file
            outfile.write(json.dumps(output_data) + '\n')
            line_count += 1

        print(f'Skipped {skipped_lines} lines due to None values.')


def convert_factcheckgpt_jsonl(input_file, output_file, num_lines):
    line_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_count, line in enumerate(infile):
            if num_lines is not None and line_count >= num_lines:
                break
            
            sample = json.loads(line)

            # Change 'prompt' key to 'question' if it exists
            if 'prompt' in sample:
                sample['question'] = sample.pop('prompt')

            # Initialize lists for claims, factuality labels, and checkworthiness
            claims = []
            gt_factuality_labels = []
            gt_checkworthiness = []

            # Iterate through sentences in the sample
            for sentence_key in sample.get('sentences', {}):
                sentence = sample['sentences'][sentence_key]
                claims.extend(sentence.get('claims', []))
                gt_factuality_labels.extend(sentence.get('claims_factuality_label', []))
                gt_checkworthiness.extend(sentence.get('claim_checkworthiness', []))

            # Calculate gt_facteval_score
            gt_facteval_score = sum(
                1 for label, check in zip(gt_factuality_labels, gt_checkworthiness) 
                if label == True and check.strip().lower() == 'factual'
            ) / len(claims) if claims else 0

            # Prepare the output data
            output_data = {
                **sample,
                'prompt_source': "factcheck-GPT",
                'model': "ChatGPT",
                "claims": claims,
                "gt_factuality_labels": gt_factuality_labels,
                "gt_checkworthiness": gt_checkworthiness,
                "gt_facteval_score": gt_facteval_score
            }

            # Write the modified sample to the output file
            json.dump(output_data, outfile)
            outfile.write('\n')
            line_count += 1


def load_data(input_file_path, output_file_path, num_lines, dataset_type):
    if "factscore" in dataset_type.lower():
        convert_factscore_jsonl(input_file_path, output_file_path, num_lines)
    elif "freshqa" in dataset_type.lower():
        # load_hf_dataset(input_file_path, output_file_path, num_lines)
        convert_freshqa_jsonl(input_file_path, output_file_path, num_lines)
    elif "longfact" in dataset_type.lower():
        convert_longfact_jsonl(input_file_path, output_file_path, num_lines)
    elif "felm" in dataset_type.lower():
        load_felm(input_file_path, output_file_path, num_lines)
    elif "expertqa" in dataset_type.lower():
        if not check_uuid_in_jsonl(input_file_path):
            add_uuid_to_jsonl_inplace(input_file_path)
        convert_expertqa_jsonl(input_file_path, output_file_path, num_lines)
    elif "factcheck-gpt" in dataset_type.lower():
        if not check_uuid_in_jsonl(input_file_path):
            add_uuid_to_jsonl_inplace(input_file_path)
        convert_factcheckgpt_jsonl(input_file_path, output_file_path, num_lines)
    else:
        print("Dataset type not applicable!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--num_lines", type=int, default=None)
    parser.add_argument("--dataset_type", type=str, required=True)
    args = parser.parse_args()

    load_data(args.input_file_path, args.output_file_path, args.num_lines, args.dataset_type)
    print(f'Completed loading {args.dataset_type}! Check {args.output_file_path}')
