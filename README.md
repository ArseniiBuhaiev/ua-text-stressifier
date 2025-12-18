# Ukrainian TTS Preprocessing

We recommend using **Python 3.10 or higher** for best compatibility.
To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Contents

* [Ukrainian Lexical Stress Prediction Model](#ukrainian-lexical-stress-prediction-model)
* [Ukrainian Phonemizer](#ukrainian-phonemizer)
* [Ukrainian Lexical Stress Benchmark](#ukrainian-lexical-stress-benchmark)
* [Wav2Vec2 with Lexical Stress](#wav2vec2-with-lexical-stress)

---

# Ukrainian Lexical Stress Prediction Model

We provide a **ByT5-based grapheme-to-phoneme model** specialized for predicting lexical stress in Ukrainian words.

### Quickstart: Predict Lexical Stress

```python
from src.accentor import UkrainianStressifier

stressifier = UkrainianStressifier()

print(stressifier.apply_stress_marks("Привіт, як у тебе справи?"))
```

### Model Highlights

* **Architecture:** ByT5 Grapheme-to-Phoneme model
* **Training Data:** Voice of America corpus, annotated with stress marks by an ASR Wav2Vec2 model

---

# Ukrainian Phonemizer

The Ukrainian Phonemizer converts Ukrainian text into phonemes.

### Usage Example

```python
from src.phonemizer import UkrainianPhonemizer

phonemizer = UkrainianPhonemizer()

print(phonemizer.phonemize("привіт світе"))
```

---

# Ukrainian Lexical Stress Benchmark

The **Ukrainian Lexical Stress Benchmark** is a manually annotated dataset created to evaluate lexical stress prediction systems in context.

**Dataset location:**

```
lexical_stress_benchmark/data/lexical_stress_dataset.csv
```

### Dataset Format

Each sentence marks stress with a `+` immediately after the stressed vowel. It contains columns:

* `StressedSentence`: Sentence with stress annotations
* `Source`: Origin (`wiki`, `plug`, or `custom`)

### Sample Entry

```csv
У+ ва+зі стоя+ли кві+ти.,custom
```

### Dataset Statistics

| Statistic                                                   | Count |
| ----------------------------------------------------------- | ----: |
| Total sentences                                             | 1,026 |
| Unique word forms (incl. inflections, derivations)          | 6,439 |
| Unique words with stress ambiguity (meaning or inflections) |   640 |
| Unique words with ≥2 stress forms in dataset                |   296 |

### Sources

* Wikipedia (300 sentences) — formal encyclopedic style
* Pluperfect GRAC (438 sentences) — fiction, journalism, poetry
* Custom (288 sentences) — manually balanced for ambiguous stress patterns

### Evaluation Metrics

* Word-Level Accuracy
* Sentence-Level Accuracy
* Unambiguous Word Accuracy
* Ambiguous Word Accuracy
* Macro-Average F1 (Ambiguous Word Pairs)

### Quickstart: Run the Benchmark

```python
from lexical_stress_benchmark.benchmark import evaluate_stressification

def custom_stressify(text):
    """
    Add '+' after the stressed vowel in each stressed word.
    """
    # your implementation here
    return text

accuracies = evaluate_stressification(custom_stressify)
for metric, value in accuracies.items():
    print(f"{metric:40} {value * 100:.2f}%")
```

---

# Wav2Vec2 with Lexical Stress

This model transcribes Ukrainian speech including lexical stress marks directly in the transcription.

* **Fine-tuned model on Hugging Face:** [mouseyy/uk\_wav2vec2\_with\_stress\_mark](https://huggingface.co/mouseyy/uk_wav2vec2_with_stress_mark)
* **Training data:** Common Voice corpus annotated with lexical stress from [Ukrainian Word Stress](https://github.com/lang-uk/ukrainian-word-stress) and [Ukrainian Accentor](https://github.com/egorsmkv/ukrainian-accentor)

---

# Citation

```
@inproceedings{senyk-etal-2025-context,
    title = "Context-Aware Lexical Stress Prediction and Phonemization for {U}krainian {TTS} Systems",
    author = "Senyk, Anastasiia  and
      Lukianchuk, Mykhailo  and
      Robeiko, Valentyna  and
      Paniv, Yurii",
    editor = "Romanyshyn, Mariana",
    booktitle = "Proceedings of the Fourth Ukrainian Natural Language Processing Workshop (UNLP 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria (online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.unlp-1.11/",
    doi = "10.18653/v1/2025.unlp-1.11",
    pages = "96--104",
    ISBN = "979-8-89176-269-5",
    abstract = "Text preprocessing is a fundamental component of high-quality speech synthesis. This work presents a novel rule-based phonemizer combined with a sentence-level lexical stress prediction model to improve phonetic accuracy and prosody prediction in the text-to-speech pipelines. We also introduce a new benchmark dataset with annotated stress patterns designed for evaluating lexical stress prediction systems at the sentence level.Experimental results demonstrate that the proposed phonemizer achieves a 1.23{\%} word error rate on a manually constructed pronunciation dataset, while the lexical stress prediction pipeline shows results close to dictionary-based methods, outperforming existing neural network solutions."
}
```
