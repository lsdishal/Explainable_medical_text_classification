import string
from typing import List, Tuple, Dict, Any

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "<unk>", "[UNK]"}

def aggregate_subwords(token_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Subword attribution aggregation layer mapping WordPiece/subword tokens 
    (e.g., 'head' + '##ache', 'se' + '##vere', 'itch' + '##ing') back into human-readable words.
    
    Attribution Combination Strategy:
    - Sum subword attribution scores for constituent subwords forming a complete word.
    - Filter Transformer special tokens ([CLS], [SEP], [PAD]) and pure punctuation.
    - Retain original raw subword tokens for research mode / technical debugging.
    """
    if not token_scores:
        return []

    word_scores: List[Tuple[str, float]] = []
    curr_word = ""
    curr_score = 0.0

    for token, score in token_scores:
        tok_str = str(token).strip()
        tok_lower = tok_str.lower()

        # Skip special tokens
        if tok_str in SPECIAL_TOKENS or tok_lower in {t.lower() for t in SPECIAL_TOKENS}:
            continue

        # Check if subword continuation
        if tok_str.startswith("##"):
            clean_sub = tok_str[2:]
            curr_word += clean_sub
            curr_score += float(score)
        elif tok_str.startswith("Ġ"):
            clean_sub = tok_str[1:]
            if curr_word:
                if not all(c in string.punctuation for c in curr_word):
                    word_scores.append((curr_word, curr_score))
            curr_word = clean_sub
            curr_score = float(score)
        else:
            if curr_word:
                if not all(c in string.punctuation for c in curr_word):
                    word_scores.append((curr_word, curr_score))
            curr_word = tok_str
            curr_score = float(score)

    if curr_word and not all(c in string.punctuation for c in curr_word):
        word_scores.append((curr_word, curr_score))

    # Strip surrounding punctuation from aggregated words (e.g. "headache." -> "headache")
    cleaned_word_scores = []
    for w, s in word_scores:
        clean_w = w.strip(string.punctuation + " ").lower()
        if clean_w:
            cleaned_word_scores.append((clean_w, s))

    # Group duplicate aggregated words (e.g., if "nausea" appears twice in input text)
    aggregated_dict = {}
    for w, s in cleaned_word_scores:
        if w not in aggregated_dict:
            aggregated_dict[w] = s
        else:
            aggregated_dict[w] += s

    return list(aggregated_dict.items())
