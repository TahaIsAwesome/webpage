import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load GPT-2 model and tokenizer once
print("Loading GPT-2 model...")
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model.eval()

def calculate_perplexity(text):
    """Calculates perplexity of a given text using GPT-2."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def average_perplexity_from_docs(doc_paths):
    """Calculates average perplexity from a list of file paths."""
    perplexities = []
    for path in doc_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        ppl = calculate_perplexity(text)
        perplexities.append(ppl)
        print(f"{os.path.basename(path)}: Perplexity = {ppl:.2f}")
    return np.mean(perplexities)

def detect_ai_by_comparison(reference_docs, target_doc, tolerance=0.7):
    """
    Compares target document perplexity to average of references.
    If it's significantly lower (more fluent), flag as likely AI-generated.
    """
    print("\nScanning reference documents...")
    avg_ref_ppl = average_perplexity_from_docs(reference_docs)

    print("\nScanning target document...")
    with open(target_doc, "r", encoding="utf-8") as f:
        target_text = f.read()
    target_ppl = calculate_perplexity(target_text)

    print(f"\nAverage reference perplexity: {avg_ref_ppl:.2f}")
    print(f"Target document perplexity: {target_ppl:.2f}")

    if target_ppl < avg_ref_ppl * tolerance:
        print("Target document is significantly more fluent — may be AI-generated.")
        return True
    else:
        print("Target document is within expected range — likely human-written.")
        return False


if __name__ == "__main__":
    reference_documents = ["ref1.txt", "ref2.txt", "ref3.txt"]
    target_document = "suspect.txt"

    tolerances = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    for tol in tolerances:
        print(f"\n--- Testing with tolerance = {tol} ---")
        is_ai = detect_ai_by_comparison(reference_documents, target_document, tolerance=tol)
        results.append("AI-generated" if is_ai else "Human-written")

    # Count the most frequent result
    from collections import Counter

    result_counts = Counter(results)
    most_common_result, count = result_counts.most_common(1)[0]

    print("\n=== Final Verdict ===")
    print(f"Based on majority vote across tolerances: {most_common_result} ({count} out of {len(tolerances)})")








# import nltk
# # nltk.download('punkt')
# #
# # from nltk.tokenize import PunktSentenceTokenizer
# #
# # def calculate_sentence_perplexities(text):
# #     tokenizer = PunktSentenceTokenizer()
# #     sentences = tokenizer.tokenize(text)
# #     # rest of your code ...
# # import nltk
# # nltk.download('punkt')
# # from nltk.tokenize import sent_tokenize


# from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt')
#
# with open("yourfile.txt", "r", encoding="utf-8") as f:
#     text = f.read()
#
# sentences = sent_tokenize(text)
# print(sentences)
# import nltk
# from nltk.tokenize import PunktSentenceTokenizer
# tokenizer = PunktSentenceTokenizer(text)
# sentences = tokenizer.tokenize(text)
# from nltk.tokenize import sent_tokenize
# sentences = sent_tokenize()
# import nltk
# nltk.download('punkt_tab')
# sent_tokenize(text, language="punkt_tab")
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
# import os
# import torch
# import numpy as np
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
#
# # Load GPT-2 model and tokenizer once
# print("Loading GPT-2 model...")
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
# model.eval()
#
# def calculate_sentence_perplexities(text):
#     """Calculates sentence-level perplexities to assess burstiness."""
#     from nltk.tokenize import sent_tokenize
#     sentences = sent_tokenize(text)
#     perplexities = []
#     for sentence in sentences:
#         inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=1024)
#         with torch.no_grad():
#             outputs = model(**inputs, labels=inputs["input_ids"])
#             loss = outputs.loss
#             ppl = torch.exp(loss)
#             perplexities.append(ppl.item())
#     return perplexities
#
# def calculate_stats(text):
#     """Calculate perplexity, burstiness, and average token surprise."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
#     with torch.no_grad():
#         outputs = model(**inputs, labels=inputs["input_ids"])
#         loss = outputs.loss
#         perplexity = torch.exp(loss).item()
#
#     sentence_ppls = calculate_sentence_perplexities(text)
#     burstiness = np.std(sentence_ppls) if sentence_ppls else 0
#
#     # Token-level surprise
#     input_ids = inputs["input_ids"]
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#     token_log_probs = log_probs[0, :-1, :].gather(1, input_ids[:, 1:].T).squeeze()
#     avg_token_surprise = -token_log_probs.mean().item()
#
#     return perplexity, burstiness, avg_token_surprise
#
# def average_stats_from_docs(doc_paths):
#     """Calculate average metrics from reference documents."""
#     all_ppl, all_burst, all_surprise = [], [], []
#     for path in doc_paths:
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
#         ppl, burst, surprise = calculate_stats(text)
#         all_ppl.append(ppl)
#         all_burst.append(burst)
#         all_surprise.append(surprise)
#         print(f"{os.path.basename(path)}: PPL={ppl:.2f}, Burst={burst:.2f}, Surprise={surprise:.2f}")
#     return np.mean(all_ppl), np.mean(all_burst), np.mean(all_surprise)
#
# def detect_ai_by_signals(reference_docs, target_doc):
#     """Compare target document metrics to reference averages."""
#     print("\nScanning reference documents...")
#     ref_ppl, ref_burst, ref_surprise = average_stats_from_docs(reference_docs)
#
#     print("\nScanning target document...")
#     with open(target_doc, "r", encoding="utf-8") as f:
#         target_text = f.read()
#     tgt_ppl, tgt_burst, tgt_surprise = calculate_stats(target_text)
#
#     print(f"\nReference avg: PPL={ref_ppl:.2f}, Burst={ref_burst:.2f}, Surprise={ref_surprise:.2f}")
#     print(f"Target metrics: PPL={tgt_ppl:.2f}, Burst={tgt_burst:.2f}, Surprise={tgt_surprise:.2f}")
#
#     ppl_flag = tgt_ppl < ref_ppl * 0.7
#     burst_flag = tgt_burst < ref_burst * 0.7
#     surprise_flag = tgt_surprise < ref_surprise * 0.9
#
#     votes = ppl_flag + burst_flag + surprise_flag
#
#     if votes >= 2:
#         print("Target document exhibits AI-like characteristics.")
#         return True
#     else:
#         print("Target document is likely human-written.")
#         return False
#
# if __name__ == "__main__":
#     reference_documents = ["ref1.txt", "ref2.txt", "ref3.txt"]
#     target_document = "suspect.txt"
#     detect_ai_by_signals(reference_documents, target_document)