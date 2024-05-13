# Bridging the Bosphorus: Advancing Turkish Large Language Models through Strategies for Low-Resource Language Adaptation and Benchmarking

## Evaluation

### Translation Evaluations

The development of Turkish question-answering datasets, TruthfulQA-TR and ARC-TR, to evaluate the reasoning capabilities of Large Language Models (LLMs) in downstream Question Answering tasks. The datasets were created by translating the TruthfulQA Multiple Choice Dataset and the ARC dataset into Turkish using the DeepL Machine Translation framework (https://github.com/DeepLcom/deepl-python). The translated samples were reviewed for errors, and the test sets from TruthfulQA-MC2 and ARC-Challenge were used for evaluations. The experiments followed the same prompting settings as LLM-Leaderboard, and the performances of all models, including open-source Turkish LLMs from Huggingface, were included in the analysis.

The original English datasets:

    arc/arc-en.jsonl
    truthfulqa/truthfulqa-en.jsonl

The final translated datasets:

    translations/arc-tr.jsonl
    translations/truthfulqa-tr.jsonl
    
Initial translations using the DeepL framework:

    arc/arc-tr.jsonl
    truthfulqa/truthfulqa-tr.jsonl

The translations are first annotated by three annotators and the annotations evaluations based on raw agreement, Cohen's and Fleiss' Kappa are calculated using the calculate_stats.py script. 

### Bits-Per-Character (BPC) evaluations

Auto-regressive language models are typically trained by optimizing the Negative Log-Likelihood (NLL) of the training data and evaluated using perplexity, a measure of prediction uncertainty. However, comparing models using different tokenizers can be challenging due to varying tokenization results. To address this, we use Bits-Per-Character (BPC), a metric derived from NLL, to evaluate performance at the character level. Our comparisons mainly rely on BPC, which normalizes the impact of tokenization differences. 

The Bits-Per-Character (BPC) for LLMs can be callculated using the script: bpc/bpc.py, which can be run with the bash file bpc/run_bpc.py. The BPC is evaluated on the trnews-64 corpus Safaya et al. (2022), comprising 5,000 samples.


