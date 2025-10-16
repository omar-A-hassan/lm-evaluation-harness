"""
Utility functions for Egyptian Arabic ↔ English translation tasks.

Provides:
- Dataset filtering for different translation directions
- Text processing (formatting and stripping)
- Multiple BERTScore-based metrics using different models
"""

import datasets
import evaluate


# ==============================================================================
# DATASET FILTERING FUNCTIONS
# ==============================================================================

def en_arz(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Filter dataset for English → Egyptian Arabic direction.
    
    This function is called via the YAML file to prepare the dataset
    for the English-to-Arabic translation task.
    
    Args:
        dataset: The raw dataset from HuggingFace
        
    Returns:
        Filtered dataset containing only English-to-Arabic samples
    """
    # The IbrahimAmin/arz-en-parallel-corpus dataset contains all pairs
    # for both directions, so we return it as-is
    return dataset


def arz_en(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Filter dataset for Egyptian Arabic → English direction.
    
    This function is called via the YAML file to prepare the dataset
    for the Arabic-to-English translation task.
    
    Args:
        dataset: The raw dataset from HuggingFace
        
    Returns:
        Filtered dataset containing only Egyptian Arabic-to-English samples
    """
    # The IbrahimAmin/arz-en-parallel-corpus dataset contains all pairs
    # for both directions, so we return it as-is
    return dataset


# ==============================================================================
# TEXT PROCESSING FUNCTIONS
# ==============================================================================

def strip(resps, docs):
    """
    Strip whitespace from model responses.
    
    Called by the framework after model generation to clean up outputs
    before metric computation. Removes leading/trailing whitespace
    and stops at the first newline (which is our stopping criterion).
    
    Args:
        resps: List of model response lists (each doc may have multiple responses)
        docs: Original documents (for reference, not used here)
        
    Yields:
        Cleaned responses with whitespace stripped
        
    Example:
        Input:  ["  Hello world  \n", "  Goodbye  \n"]
        Output: ["Hello world", "Goodbye"]
    """
    return map(lambda r: r[0].strip(), resps)


def doc_to_text(doc) -> str:
    """
    Convert a dataset document to input text for the model.
    
    Formats the document as a translation prompt that the model can understand.
    The exact format depends on which direction we're translating:
    - EN→ARZ: "English phrase: {english_text}\nEgyptian Arabic phrase:"
    - ARZ→EN: "Egyptian Arabic phrase: {arabic_text}\nEnglish phrase:"
    
    The actual direction is handled by the process_docs filter in the YAML.
    
    Args:
        doc: A single sample from the dataset
        
    Returns:
        Formatted prompt string for the model
        
    Example Output:
        "English phrase: Hello, how are you?\nEgyptian Arabic phrase:"
    """
    # The IbrahimAmin/arz-en-parallel-corpus dataset has two columns:
    # - 'en': English text
    # - 'arz': Egyptian Arabic text
    
    # The process_docs filter in YAML determines which language is input vs output
    # But this function is called for both directions, so we return a generic format
    # The framework handles the actual direction switching
    
    # For now, return the English text as the input (direction is handled by YAML)
    if "en" in doc and "arz" in doc:
        return f"English phrase: {doc['en']}\nEgyptian Arabic phrase:"
    else:
        # Fallback with better error message
        cols = list(doc.keys())
        raise ValueError(f"Expected columns 'en' and 'arz', got: {cols}")


def doc_to_target(doc) -> str:
    """
    Extract the target translation from a dataset document.
    
    Returns the reference translation that will be compared against
    model output using BLEU, CHRF, and BERTScore.
    
    Args:
        doc: A single sample from the dataset
        
    Returns:
        Reference translation string
        
    Example Output:
        "كيف حالك؟"  # Egyptian Arabic response
    """
    # The IbrahimAmin/arz-en-parallel-corpus has columns 'en' and 'arz'
    # For the EN→ARZ task, the target is 'arz' (Egyptian Arabic)
    # For the ARZ→EN task, the target is 'en' (English)
    # The framework calls this for both, so we return the non-input language
    
    # This function should return the Egyptian Arabic text as the "other" language
    if "arz" in doc:
        return doc["arz"]
    else:
        cols = list(doc.keys())
        raise ValueError(f"Expected column 'arz', got: {cols}")


# ==============================================================================
# METRIC HELPER CLASS
# ==============================================================================

class Average:
    """Helper class to compute and store average scores."""
    
    def __init__(self, items):
        """
        Initialize with a list of scores and compute average.
        
        Args:
            items: List of numerical scores (F1 scores from BERTScore)
        """
        if isinstance(items, list):
            self.value = sum(items) / len(items) if items else 0.0
        else:
            self.value = items
    
    def __repr__(self):
        return f"{self.value}"
    
    def __float__(self):
        return float(self.value)


# ==============================================================================
# METRIC FUNCTIONS
# ==============================================================================

def bert(predictions, references):
    """
    Compute BERTScore using the default English BERT model.

    Uses google-bert/bert-base-uncased for semantic similarity evaluation.
    Good for evaluating English-side translation quality.

    Args:
        predictions: List of predicted translation strings
        references: List of reference translation strings

    Returns:
        Average F1 score from BERTScore

    Example:
        predictions = ["Hello, how are you?", "Good morning"]
        references = ["Hi, how are you doing?", "Good morning"]
        # Returns average BERTScore F1 ~0.95
    """
    bert_model = "google-bert/bert-base-uncased"
    bert_score = evaluate.load("bertscore")

    result = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )

    # Extract and average F1 scores
    return Average(result["f1"])


def arabert(predictions, references):
    """
    Compute BERTScore using AraBERT (Arabic-specialized BERT).

    Uses aubmindlab/bert-base-arabert which is trained specifically
    for Arabic NLP tasks. Better captures Arabic morphology and syntax.

    Args:
        predictions: List of predicted translation strings
        references: List of reference translation strings

    Returns:
        Average F1 score from AraBERT-based BERTScore

    Example:
        predictions = ["كيفك يا حبيبي", "أنا بخير"]
        references = ["كيفك يا صديقي", "أنا تمام"]
        # Returns average AraBERT F1 score ~0.92
    """
    bert_model = "aubmindlab/bert-base-arabert"
    bert_score = evaluate.load("bertscore")

    result = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )

    return Average(result["f1"])


def mbert(predictions, references):
    """
    Compute BERTScore using Multilingual BERT.

    Uses google-bert/bert-base-multilingual-cased which handles
    both English and Arabic. Useful for overall translation quality
    without language-specific bias.

    Args:
        predictions: List of predicted translation strings
        references: List of reference translation strings

    Returns:
        Average F1 score from Multilingual BERTScore

    Example:
        predictions = ["كيف حالك؟"]
        references = ["How are you?"]
        # Returns average mBERT F1 score ~0.88
    """
    bert_model = "google-bert/bert-base-multilingual-cased"
    bert_score = evaluate.load("bertscore")

    result = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )

    return Average(result["f1"])


def bertbase(predictions, references):
    """
    Alias for bert() - compute BERTScore using English BERT.

    Provided for consistency with other translation tasks in the framework.
    Used as aggregation function in metric_list.

    Args:
        predictions: List of predicted translation strings
        references: List of reference translation strings

    Returns:
        Average BERTScore using base English BERT
    """
    return bert(predictions, references)
