# Egyptian Arabic ↔ English Translation Tasks

This module provides machine translation evaluation tasks for translating between Egyptian Arabic (dialect) and English in both directions.

## Overview

The Egyptian Arabic-English translation tasks enable evaluation of machine translation models on the low-resource Egyptian Arabic dialect, using a professionally curated parallel corpus. These tasks support:

- **English → Egyptian Arabic** (`en_arz`)
- **Egyptian Arabic → English** (`arz_en`)
- **Combined task** (`arz_en_all`) - evaluates both directions together

## Dataset

**Source:** [IbrahimAmin/arz-en-parallel-corpus](https://huggingface.co/datasets/IbrahimAmin/arz-en-parallel-corpus)

**Statistics:**
- **Total pairs:** 26,851 sentence pairs
- **Training set:** 25,000 pairs
- **Test set:** 1,851 pairs
- **Quality:** Professionally curated, deduplicated
- **Merge source:** Multiple high-quality Egyptian Arabic corpora

**Languages:**
- **Egyptian Arabic** (code: `arz`) - Modern Standard Arabic dialect spoken in Egypt
- **English** (code: `en`)

## Evaluation Metrics

Each translation task is evaluated using six complementary metrics:

### 1. **BLEU (Bilingual Evaluation Understudy)**
- **Range:** 0-100 (higher is better)
- **Characteristics:** Precision-based n-gram overlap metric, widely used for machine translation
- **Use case:** Overall translation adequacy

### 2. **TER (Translation Edit Rate)**
- **Range:** 0-100 (lower is better)
- **Characteristics:** Edit distance normalized by average reference length
- **Use case:** Quantifies minimum edits needed to achieve reference
- **Higher is worse** for this metric (configured appropriately in aggregation)

### 3. **CHRF (Character F-score)**
- **Range:** 0-100 (higher is better)
- **Characteristics:** Character-level n-gram overlap, robust to morphological variation
- **Use case:** Captures character-level translation quality, especially important for Arabic morphology

### 4. **BERTScore (English BERT)**
- **Model:** `google-bert/bert-base-uncased`
- **Range:** 0-1 (higher is better)
- **Use case:** Semantic similarity of predictions vs. references using English representations
- **Best for:** English-side quality assessment

### 5. **AraBERTScore**
- **Model:** `aubmindlab/bert-base-arabert`
- **Range:** 0-1 (higher is better)
- **Characteristics:** Arabic-specialized BERT, captures Arabic morphology and syntax
- **Use case:** Semantic quality assessment that understands Arabic linguistic structure
- **Best for:** Arabic translation quality and dialect representation

### 6. **Multilingual BERTScore**
- **Model:** `google-bert/bert-base-multilingual-cased`
- **Range:** 0-1 (higher is better)
- **Use case:** Cross-lingual semantic similarity without language bias
- **Best for:** Overall translation coherence across languages

## Tasks

### Task: `en_arz` - English to Egyptian Arabic

**Configuration:** `arz_en_translation_en_arz.yaml`

Translates English source text to Egyptian Arabic target text.

**Example:**
```
Input: "Hello, how are you today?"
Reference: "السلام عليكم، كيفك انهاردة؟"
```

### Task: `arz_en` - Egyptian Arabic to English

**Configuration:** `arz_en_translation_arz_en.yaml`

Translates Egyptian Arabic source text to English target text.

**Example:**
```
Input: "كيفك يا حبيبي؟"
Reference: "How are you, my dear?"
```

### Task: `arz_en_all` - Combined Bidirectional

**Configuration:** `arz_en_translation_all.yaml`

Evaluates model on both translation directions in a single evaluation run.

## Usage

### Running Individual Tasks

```bash
# Evaluate English → Egyptian Arabic
python -m lm_eval --model gpt2 --tasks en_arz

# Evaluate Egyptian Arabic → English
python -m lm_eval --model gpt2 --tasks arz_en
```

### Running Combined Task

```bash
# Evaluate both directions
python -m lm_eval --model gpt2 --tasks arz_en_all
```

### Using Specific Model

Example with the NLLB-200 distilled model (recommended for this task):

```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=facebook/nllb-200-distilled-600M \
    --tasks arz_en \
    --num_fewshot 0 \
    --batch_size auto
```

## Implementation Details

### File Structure

```
lm_eval/tasks/arz_en_translation/
├── utils.py                           # Utility functions
├── arz_en_common_yaml                # Shared configuration
├── arz_en_translation_en_arz.yaml    # English → Arabic task
├── arz_en_translation_arz_en.yaml    # Arabic → English task
├── arz_en_translation_all.yaml       # Combined bidirectional task
└── README.md                          # This file
```

### Utility Functions

**Dataset Filtering:**
- `en_arz(dataset)` - Prepares dataset for EN→ARZ direction
- `arz_en(dataset)` - Prepares dataset for ARZ→EN direction

**Text Processing:**
- `doc_to_text(doc)` - Formats dataset sample into model prompt
- `doc_to_target(doc)` - Extracts reference translation
- `strip(resps, docs)` - Cleans model responses for metric computation

**Metric Functions:**
- `bert(items)` - Computes BERTScore using English BERT
- `arabert(items)` - Computes BERTScore using AraBERT
- `mbert(items)` - Computes BERTScore using Multilingual BERT
- `bertbase(items)` - Alias for `bert()` (used in aggregation)
- `Average(items)` - Helper class for averaging metric scores

## Model Recommendations

### NLLB-200 Series

The **NLLB-200 (No Language Left Behind)** models are highly recommended for this task:

**Supported variants:**
- `facebook/nllb-200-distilled-600M` ⭐ **Recommended** (fast, good quality)
- `facebook/nllb-200-distilled-1.3B` (higher quality)
- `facebook/nllb-200-1.3B` (higher quality, larger)
- `facebook/nllb-200-3.3B` (best quality, larger)

**Language codes in NLLB:**
- Egyptian Arabic: `arz_Arab`
- English: `eng_Latn`

**Example evaluation command:**

```bash
python -m lm_eval \
    --model hf \
    --model_args "pretrained=facebook/nllb-200-distilled-600M,dtype=float16" \
    --tasks en_arz,arz_en \
    --num_fewshot 0 \
    --batch_size 8
```

### Other Supported Models

Any model supporting sequence-to-sequence translation can be used:
- **mT5 series** (`google/mt5-base`, `google/mt5-large`)
- **mBART models** (`facebook/mbart-large-50-one-to-many-mmt`)
- **Custom fine-tuned models**

## Citation

If you use this task in your research, please cite:

```bibtex
@article{hassan2023arz,
  title={IbrahimAmin/arz-en-parallel-corpus},
  author={Ibrahim Amin},
  year={2023},
  howpublished={\url{https://huggingface.co/datasets/IbrahimAmin/arz-en-parallel-corpus}}
}

@article{costa-jussa2022nllb,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={Costa-jussà, Marta R. and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}

@inproceedings{lm-eval-harness,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and others},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

## References

- [IbrahimAmin/arz-en-parallel-corpus on HuggingFace](https://huggingface.co/datasets/IbrahimAmin/arz-en-parallel-corpus)
- [NLLB-200 Models](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [AraBERT - Arabic BERT](https://github.com/aribowling/AraBERT)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)

## Support and Feedback

For issues, questions, or feedback about these translation tasks, please refer to the main lm-evaluation-harness repository and documentation.
