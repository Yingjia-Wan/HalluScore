# get K median and K max
from collections import defaultdict
import csv
import re
import regex
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize

def clean_claim_label(claim, suffixes):
    """
    Clean the claim by removing any of the specified suffixes if the claim ends with one of them.
    
    Args:
        claim (str): The claim to clean.
        suffixes (list): A list of suffixes to check against.
        
    Returns:
        str or None: The cleaned claim if it matches a suffix, otherwise None.
    """
    for suffix in suffixes:
        if claim.endswith(suffix):
            return claim.replace(suffix, "").strip()  # Clean the claim and return it
    return None  # If no suffix matches, return None


def is_abstain_response(response):
    abstain_responses = [
        "i am sorry",
        "i'm sorry",
        "sorry, but i can't",
        "sorry, but i cannot",
        "sorry, but i can not",
        "sorry, i can't",
        "sorry, i can not",
        "sorry, i cannot",
        "sorry, i do not",
        "sorry, i don't",
        "sorry, i am not",
        "sorry, i'm not"
    ]
    normalized_response = response.lower()
    # if model abstains from answering or generate empty response:
    if normalized_response.strip() == '' or any(phrase in normalized_response for phrase in abstain_responses):
        return True
    return False

def clean_claim_format(response):
    # remove "Facts:" or "Claims:" line
    response = re.sub(r'^Facts:|^Claims:', '', response, flags=re.IGNORECASE | re.MULTILINE).strip()
    # remove itemized list
    claims = [x.strip().replace("- ", "") for x in response.split("\n")]
    # remove numbers in the beginning
    claims = [regex.sub(r"^\d+\.?\s", "", x) for x in claims]
    # Filter out any empty strings
    claims = [claim for claim in claims if claim]
    # remove <SOS> and <EOS> if exists
    claims = [re.sub(r'<SOS>|<EOS>', '', claim).strip() for claim in claims if claim]
    return claims

def check_token_logprobs(claim, logprobs_data):
    '''
    returns the logprob for the label token.
    if logprobs_data has not been collected or if no T/F label is found, return 0 -> it always passes the threshold.
    '''
    # if no logprobs data from the model:
    if logprobs_data is None:
        return 0
        
    # Extract the token to check (either "TRUE" or "FALSE")
    match = re.search(r'###(TRUE|FALSE)###', claim)
    if not match:
        return 0  # Return 0 if no match is found
    target_token = match.group(1)
    
    # find the logprob of the target token
    for token, logprob in zip(logprobs_data['tokens'], logprobs_data['logprob']):
        if token == target_token:
            return logprob
    return 0  # Return 0 if token not found


def get_device_map():
    import torch
    """Determine the device map based on available GPUs or fallback to CPU."""
    if torch.cuda.is_available():
        return "auto"  # Automatically use available GPUs
    else:
        return {"": "cpu"}  # Use CPU

############################################## BM25 Retriever ##############################################
def split_into_chunks(text, target_chunk_size=200, overlap=50):
    """
    Split a document into chunks with complete sentences and rough size constraints.
    Args:
        text (str): The document text to split.
        target_chunk_size (int): The target size of each chunk (in words).
        overlap (int): The number of overlapping words between chunks.
    Returns:
        list: List of text chunks.
    """


    sentences = sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = []
    current_word_count = 0

    # Chunk Splitting while respecting sentence boundaries
    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If adding this sentence exceeds the target chunk size (and the chunk is not empty), finalize the chunk
        if current_word_count + sentence_word_count > target_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap
            overlap_words = " ".join(current_chunk[-overlap:]).split() if overlap > 0 else []
            current_chunk = overlap_words + [sentence]
            current_word_count = len(overlap_words) + sentence_word_count
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def retrieve_relevant_passages(claim, evidence_list, target_chunk_size=200, overlap=50, n=3):
    """
    Retrieve relevant chunks from each document in evidence_list using BM25.
    Args:
        claim (str): The claim to match against.
        evidence_list (list): List of evidence dictionaries (each containing 'text').
        target_chunk_size (int): The target size of each chunk (in words).
        overlap (int): The number of overlapping words between chunks.
        n (int): Number of top chunks to retrieve per document.
    Returns:
        list: List of evidence dictionaries with an additional 'retrieved_chunks' field.
    """
    # Tokenize the claim
    tokenized_claim = claim.split()

    # Process each evidence document
    for ev in evidence_list:
        if 'text' not in ev:
            ev['retrieved_text'] = [ev['description']] if 'description' in ev else []
        elif 'warning' in ev or not ev['text'].strip():
            ev['retrieved_text'] = [ev['description']]
        else:
            # Split the document into chunks
            chunks = split_into_chunks(ev['text'], target_chunk_size=target_chunk_size, overlap=overlap)
            
            # Tokenize each chunk
            tokenized_chunks = [chunk.split() for chunk in chunks]
            
            # Initialize BM25 for this document
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Get the top n relevant chunks for the claim
            top_indices = bm25.get_top_n(tokenized_claim, range(len(chunks)), n=n)
            retrieved_chunks = [chunks[i] for i in top_indices]
            
            # Add the retrieved chunks to the evidence dictionary
            ev['retrieved_text'] = retrieved_chunks

    return evidence_list

############################################## Veriscore Calculation ##############################################
def get_K_stats(domain_model_triplet_dict):
    domain_K_dict = defaultdict(lambda: defaultdict(int)) # initiate a nested dict
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        claim_num_lst = []
        for model_name, triplet_lst in model_triplet_dict.items():
            for triplet in triplet_lst: # triplet: [
                                    # number of all supported claims (i.e., support score), 
                                    # number of all claims, 
                                    # number of sentences
                                    # ]
                claim_num_lst.append(triplet[1])

        claim_num_lst.sort() # a list of extracted claims count
        K_median = claim_num_lst[len(claim_num_lst)//2]
        K_max = claim_num_lst[-1]
        domain_K_dict[domain]["K_median"] = K_median
        domain_K_dict[domain]["K_max"] = K_max
        # print(f"get_K_stats: {domain} - {K_median}: {K_max}")
    # print(domain_K_dict)
    return domain_K_dict

def write_avg_numbers(domain_model_triplet_dict, domain_model_response_stats_dict, domain_K_dict, 
                      result_dir, 
                      model_name_extraction, model_name_verification,
                      total_prompt_tok_cnt, total_resp_tok_cnt, total_time,
                      abstain_count, sample_size):
    # open the score output csv file, and write the header row
    with open(result_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        # header
        writer.writerow(["model_name_extraction", "model_name_verification", 
                         "Evaled Model", "Domain", "Sample Size", "Abstain Rate", 
                         "Total Prompt Tokens", "Total Resp Tokens", "Total Time",
                         "Avg Sents", "All claims", "Presupport", "Preunsupport", "Verify", "All Support claims",
                         "Pre-verification Rate", "Pre-support rate in Pre-verification", "Pre-support rate in All claims",
                         "P", 
                         "K Med", "K Max",
                         "Rec Med", "Rec Max", 
                         "F1 Med", "F1 Max"])

        # calculate abstain rate:
        abstain_rate = abstain_count / sample_size

        # calculate avg score per domain per model; each domain and model should make an independent row.
        for domain, model_triplet_dict in domain_model_triplet_dict.items():

            K_median = domain_K_dict[domain]['K_median']
            K_max = domain_K_dict[domain]['K_max']

            for model_name in model_triplet_dict.keys():
                
                triplet_lst = domain_model_triplet_dict[domain][model_name] # # triplet: [
                                                                                            # number of all supported claims (i.e., support score), 
                                                                                            # number of all claims, 
                                                                                            # number of sentences
                                                                                            # ]
                response_stats_lst = domain_model_response_stats_dict[domain][model_name] #     [{
                                                                                                    # "sentences": 1,
                                                                                                    # "verified_claims": 2,
                                                                                                    # "verified_supported_claims": 2,
                                                                                                    # "pre_supported_claims": 4,
                                                                                                    # "pre_verified_claims": 6,
                                                                                                    # "all_claims": 6
                                                                                                    # }]

                preverified_claims_lst = [x["pre_verified_claims"] for x in response_stats_lst]
                presupported_claims_lst = [x["pre_supported_claims"] for x in response_stats_lst]
                preunsupported_claims_lst = [verified - supported for verified, supported in zip(preverified_claims_lst, presupported_claims_lst)]

                sent_len_lst = [x[2] for x in triplet_lst]
                sup_lst = [x[0] for x in triplet_lst]
                uns_lst = [x[1] - x[0] for x in triplet_lst]
                all_lst = [x[1] for x in triplet_lst]

                prec_lst = [x[0] / x[1] if x[1] != 0 else None for x in triplet_lst]
                rec_med_lst = [min(x[0] / K_median, 1) if K_median != 0 else None for x in triplet_lst]
                rec_max_lst = [min(x[0] / K_max, 1) if K_max != 0 else None for x in triplet_lst]

                # filtering out None values
                filtered_prec_lst = [x for x in prec_lst if x is not None]
                filtered_rec_med_lst = [x for x in rec_med_lst if x is not None]
                filtered_rec_max_lst = [x for x in rec_max_lst if x is not None]

                # get f1@K median and f1@K max
                f1_med_lst = [2 * prec * rec_med / (prec + rec_med) if rec_med > 0 else 0 for prec, rec_med in zip(filtered_prec_lst, filtered_rec_med_lst)]
                f1_max_lst = [2 * prec * rec_max / (prec + rec_max) if rec_max > 0 else 0 for prec, rec_max in zip(filtered_prec_lst, filtered_rec_max_lst)]

                # get avg. numbers
                ave_sent = sum(sent_len_lst) / len(sent_len_lst)
                avg_preverified_claims = sum(preverified_claims_lst) / len(preverified_claims_lst)
                avg_presupported_claims = sum(presupported_claims_lst) / len(presupported_claims_lst)
                avg_preunsupported_claims = sum(preunsupported_claims_lst) / len(preunsupported_claims_lst)

                S = sum(sup_lst) / len(sup_lst)
                U = sum(uns_lst) / len(uns_lst)
                All_claims = sum(all_lst) / len(all_lst)
                preverification_rate = avg_preverified_claims / All_claims if All_claims != 0 else None
                presupport_rate_in_preverification = avg_presupported_claims / avg_preverified_claims if avg_preverified_claims != 0 else None
                presupport_rate_in_all_claims = avg_presupported_claims / All_claims if All_claims != 0 else None

                P = sum(filtered_prec_lst) / len(filtered_prec_lst) if len(filtered_prec_lst) != 0 else None
                Rec_med = sum(filtered_rec_med_lst) / len(filtered_rec_med_lst) if len(filtered_rec_med_lst) != 0 else None
                Rec_max = sum(filtered_rec_max_lst) / len(filtered_rec_max_lst) if len(filtered_rec_max_lst) != 0 else None
                F1_med = sum(f1_med_lst) / len(f1_med_lst) if len(f1_med_lst) != 0 else None
                F1_max = sum(f1_max_lst) / len(f1_max_lst) if len(f1_max_lst) != 0 else None


                # Check for None values and replace them with "N/A" or another placeholder
                P = round(P, 3) if P is not None else "N/A"
                Rec_med = round(Rec_med, 3) if Rec_med is not None else "N/A"
                Rec_max = round(Rec_max, 3) if Rec_max is not None else "N/A"
                F1_med = round(F1_med, 3) if F1_med is not None else "N/A"
                F1_max = round(F1_max, 3) if F1_max is not None else "N/A"

                table_row = [model_name_extraction, model_name_verification, 
                            model_name, domain, sample_size, abstain_rate,
                            total_prompt_tok_cnt, total_resp_tok_cnt, total_time,
                            round(ave_sent, 3), round(All_claims, 3), round(avg_presupported_claims,3), round(avg_preunsupported_claims, 3),round(avg_preverified_claims, 3), round(S, 3),  
                            preverification_rate, presupport_rate_in_preverification, presupport_rate_in_all_claims,
                            P, 
                            K_median, K_max,
                            Rec_med, Rec_max, 
                            F1_med, F1_max]

                writer.writerow(table_row)
                # print(f"[{domain}-{model_name}] \nF1@k median: {F1_med:.3f}, F1@k max: {F1_max:.3f}, Precision: {P:.3f}")
                print(f"[{domain}-{model_name}] \nF1@k median: {F1_med}, F1@k max: {F1_max}, Precision: {P}")

        # print(f"Average score saved to {result_dir}!")

def get_halluscore(domain_model_triplet_dict, domain_model_response_stats_dict,
                  result_dir, 
                  model_name_extraction, model_name_verification, 
                  total_prompt_tok_cnt, total_resp_tok_cnt, total_time,
                  abstain_count, sample_size):
    # get stats for all domains and models
    domain_K_dict= get_K_stats(domain_model_triplet_dict)
    write_avg_numbers(domain_model_triplet_dict, domain_model_response_stats_dict, domain_K_dict, 
                      result_dir, 
                      model_name_extraction, model_name_verification,
                      total_prompt_tok_cnt, total_resp_tok_cnt, total_time,
                      abstain_count, sample_size)