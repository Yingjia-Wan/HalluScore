
# HalluScore

HalluScore is a holistic factuality evaluation metric of long-form generations, focusing on two dimensions of facuality: **factual precision** and **factual density**. The evaluation pipeline follows the decompose-then-verify framework which consists of three steps: (1) `claim extraction`, (2) `evidence collection`, and (3) `claim verification`. However, HalluScore offers the following major features addressing several pain points:

- **Efficient**: HalluScore adds a simple 'pre-verificatoin' task to the `claim extraction` stage during the LLM extractor's generation, drastically improving the compute efficiency and high token usages. This also lifts the burden of requirements for time-consuming and menually engineered claim revisions. In addition, asynchronous multi-step processing are implemented to optmize the pipeline efficiency.
- **Uncertainty Checking**: To systematically curb the LLM-as-judge's overconfidence during pre-verification and `claim verification`, HalluScore calibrates a *token-logprob-based threshold* as a proxy for uncertainty checking, so as to automatically check the validity of each verification label.
- **Domain-Agnostic**: HalluScore is applicable to all QA/non-QA generations (e.g., knowledge-based QA, story-writing, reasoning), and can especially cater to evaluating long-form generation. [TO ADD]
- **Reliable**: By collecting and scraping *document-level long-context evidence* from the open web, HalluScore enables a more powerful *retrieval-augmented verifier* equipped with a dynamic and rich search-augmented knowledge source, addressing the frequent issues of irrelevant evidence/inconclusive verification. The long-context evidene can also provide the verifier with the reference source to justify its verification by locating relevant texts, offering more transparency to the decompose-then-verify evaluation framework.
- **Factual Precision vs. Factual Density**:

*Under-Construction Note:* The repo currently provides a easy-to-run metric tool for factuality evaluation of any QA generation. Feel free to try it out. More details about score calculation, benchmark results, and human evaluation will be provided later.

## Repository Structure
```
HalluScore
├── data
│   ├── data_sample.jsonl
│   ├── data_sample2.jsonl
├── halluscore
│   ├── __init__.py
│   ├── claim_extractor.py
│   ├── claim_verifier.py
│   ├── estimate_tokens_cost.py
│   ├── halluscore.py
│   ├── response_API.py
│   ├── utils.py
│   ├── web_search_API.py
├── prompt
│   ├── extraction
│   └── verification
└── requirements.txt
```

## Setup
1. Innitialize a new Python 3.9+ environment using `virtualenv` or `conda`.
2. Install the requirements.
3. Download `en_core_web_sm` using `spacy` library
    ```
    conda create --name [YOUR CONDA ENV NAME] python=3.9
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
4. Add an OpenAI or Claude API key as an environment variable in [/halluscore/.env](/halluscore/.env) for claim extractiona and claim verifictaion; then add a Jina Reader API key as well. The latter is used in the evidence colletion step for searching and scraping from the open web. Get the free API key [here](https://jina.ai/reader). Alternatively, you can replace the modules in [/halluscore/web_search_API.py](/halluscore/web_search_API.py) with your own searching API and scraping tools.
    ```
    OPENAI_API_KEY=[YOU OPENAI KEY]
    OPENAI_BASE_URL=[YOU OPENAI BASE_URL (if not default)]
    JINA_KEY=[YOUR JINA READER KEY]
    ```


## Running HalluScore
This is an end-to-end pipeline for running HalluScore on a input file containing long-form QA generations.

### Example:
```
python3 -m halluscore.halluscore \
  --data_dir ./data \
  --input_file data_sample.jsonl \
  --model_name_extraction gpt-4o-mini \
  --model_name_verification gpt-4o-mini \
  --label_n 3 \
  --pre_veri_label_m 5 \
  --extraction_method chunk \
  --ignore_cache
```
### Arguments:

* `data_dir`: Directory containing input data. `./data` by default.
* `input_file`: Name of the input data file. Two sample input files are provided in [./data](./data) (data_sample.jsonl and data_sample2.jsonl). It should be in the `jsonl` format where each json line contains
    * `question`: A query to prompt a language model for an output
    * `response`: An output generated by the language model given the `question`
    * `model`: Name of the model that generated the response
    * `prompt_source`: Name of the dataset from where the `question` is from (e.g., FreshQA)
* `model_name_extraction`: Name of the model used for claim extraction; `gpt-4-0125-preview` by default.
* `model_name_verification`: Name of the model used for claim verification; `gpt-4o` by default.
* `ignore_cache`: If specified, ignores cached results and recomputes everything. False by default.

### Other optional arguments:

* `extraction_method`: Method used for extracting claims from the response. Choices are `chunk` and `sliding_window`.
    * `chunk`: Divides the response into chunks.
    * `sliding_window`: Uses a sliding window to extract claims with context. (context1 = 0-3 sentence) <SOS>Sentence to be focused on<EOS> (context2 = 0-1 sentence)
* `stride`: You can specify a fixed stride in chunking; `0` means feeding the whole response for extraction; `-1` means dynamic stride based on response length.
* `search_res_num`: The number of evidence results to search for and save. `5` by default.
* `verify_res_num`: The number of evidence results used for verification. `5` by default.
* `label_n`: This is the type of label for claim verification. It could be `2` (binary) or `3` (ternary):
    * `2`: `supported` and `unsupported`.
    * `3`: `supported`, `contradicted`, and `inconclusive`.
* `pre_veri_label_m`: The number of labels used in the pre-verification step. Can be `3` or `5`.
* `do_not_pre_verify`: If specified, skips the pre-verification step. False by default.
* `logprob_threshold`: The log probability threshold for filtering extractions. Defaults to negative infinity (`-inf`).
* `use_external_extraction_model`: If specified, it uses your custom model instead of the one from the API call. We use Unsloth for the fine-tuned model. False by default.
* `use_external_verification_model`: If specified, it uses your custom model instead of the one from the API call. We use Unsloth for the fine-tuned model. False by default.
* `use_base_extraction_model`: If specified, it uses an open-source model for extraction. False by default.
* `use_base_verification_model`: If specified, it uses an open-source model for verification. False by default.


### Exampled output to be saved:
*   An output folder called "data_sample/chunk_m=5_gpt-4o-mini_gpt-4o-mini/". This output folder name is constructed as follows: `{input_filename}/{extraction_method}_m={pre_veri_label_m}_{model_name_extraction}_{model_name_verification}`.

    Within this folder, you'll find the following files:

    *   **`claims.jsonl`:** Contains the extracted claims from the input responses. Each line in this JSONL file represents a single claim.
    *   **`retrieved_evidence.jsonl`:** Contains the search results (evidence) retrieved for each extracted claim. Each line includes (1) the raw web-page documental evidence and (2) the retrieved evidence for each claim `claims.jsonl`
    *   **`verification_label_n=3.jsonl`:** Contains the verification results for each claim. The file name includes the `label_n` value. Each line provides the verification label (e.g., supported, contradicted, inconclusive) for each claim in `claims.jsonl`.
    *   **`veriscore_label_n=3`:** Contains the calculated average HalluScore. The file name includes the `label_n` value. This file contains a single floating-point number representing the average HalluScore across all verified claims.


## Benchmark Results

## Human Evaluation

To be updated.
