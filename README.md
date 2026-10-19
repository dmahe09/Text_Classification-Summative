# Hausa News Topic Classification

Classify Hausa news articles into one of seven topics, comparing classical, recurrent and Transformer models, and serve the best one as a web app.

**Individual capstone project, Text Classification for an African language, October 2026**
**Author:** MAHE Digne · **Course:** ML Techniques I

## Project links

| Resource | Link |
|---|---|
| Live system (Gradio app) | https://b0dee0b6cc76966290.gradio.live/ |
| Demo video |    (https://drive.google.com/drive/folders/1VSV1RG_5_5GSGIMpMCwc9x0Le2Pro474) |
| Fine-tuned model (Hugging Face Hub) | https://huggingface.co/dmahe09/hausa-news-classifier    |

> **About the live system:** the app is run from a Google Colab notebook (`run_app_colab.ipynb`) using a temporary public Gradio link, so the link expires and changes each time the notebook is started. See [Run the app](#run-the-app) to launch your own copy in a few minutes. The demo video shows the app working. *(Edit this note if you host it somewhere permanent.)*

## Contents

1. [Overview](#overview)
2. [Results at a glance](#results-at-a-glance)
3. [Dataset](#dataset)
4. [Methods](#methods)
5. [All experiments](#all-experiments)
6. [Error analysis](#error-analysis)
7. [Repository structure](#repository-structure)
8. [Reproduce the experiments](#reproduce-the-experiments)
9. [Run the app](#run-the-app)
10. [Limitations](#limitations)
11. [Resources used and what I built myself](#resources-used-and-what-i-built-myself)
12. [Licences and acknowledgements](#licences-and-acknowledgements)
13. [References](#references)

## Overview

**Problem.** Given a Hausa news headline and article text, predict its topic: `business`, `entertainment`, `health`, `politics`, `religion`, `sports` or `technology`.

**Why it matters.** African languages have far fewer NLP tools and labelled datasets than high-resource languages. A Hausa topic classifier helps organise news archives and supports newsrooms, researchers and developers who work with Hausa text.

**What I did.**
- Explored the Hausa part of the MasakhaNEWS dataset (size, class balance, text length, leakage, vocabulary).
- Built a strong baseline (TF-IDF + Logistic Regression) and ran 17 further experiments: other sparse models, BiLSTM/BiGRU with Word2Vec, and fine-tuned Transformers (AfroXLMR, AfriBERTa, XLM-R).
- Chose the best model using the validation set only, then evaluated once on the test set.
- Analysed the errors and deployed the selected model with Gradio.

## Results at a glance

The baseline is a TF-IDF (word 1–2 grams) + Logistic Regression model (C = 1). The selected model is **AfroXLMR-base, fully fine-tuned with up to 512 tokens** (experiment T5), chosen because it had the highest validation macro-F1.

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Baseline S1: TF-IDF + LogReg (C=1) | 0.881 | 0.871 |
| Best tuned sparse model S2: TF-IDF + LogReg (C=10) | 0.893 | 0.885 |
| Best RNN N2: BiLSTM + Word2Vec | 0.896 | 0.888 |
| **Selected model T5: AfroXLMR-base, len 512** | **0.914** | **0.908** |

- T5 vs the baseline: the models disagreed on 71 test articles; T5 was right on 46 of them and the baseline on 25 (exact McNemar test, p = 0.017).
- The largest gains are in the weakest classes (test F1, baseline → T5): business 0.759 → 0.844, technology 0.760 → 0.850.
- Africa-focused pretraining matters: AfroXLMR scored 0.903 vs 0.867 for generic XLM-R under the same settings (T4 vs T1).
- Full fine-tuning is necessary: a frozen AfroXLMR encoder scored only 0.808.
- Sequence length (128/256/512) and learning rate (2e-5/5e-5) made little difference (0.903–0.908).
- AfriBERTa (T2) scored 0.910 on the test set, marginally higher than T5 (0.908). T5 was still selected because the selection rule used validation scores only. The fine-tuned African-language models are within about 0.7 points of each other, which is within noise for a 637-article test set with a single run.

![Per-class F1, baseline vs T5](figures/fig_per_class_f1.png)

![All experiments compared](figures/fig_experiment_comparison.png)

## Dataset

- **Source:** the Hausa portion of **MasakhaNEWS** (Adelani et al., 2023), distributed under AFL-3.0 via the Masakhane organisation on Hugging Face (`masakhane/masakhanews`). The data is real news text; no synthetic data was used.
- **Files used:** `train.tsv`, `dev.tsv`, `test.tsv`, `labels.txt` (the notebook expects exactly these). Each article has a headline, body text, URL and category.
- **Size:** 3,173 articles in total.

| Split | Articles | business | entertainment | health | politics | religion | sports | technology |
|---|---|---|---|---|---|---|---|---|
| Train | 2,219 | 279 | 350 | 345 | 350 | 345 | 347 | 203 |
| Dev | 317 | 40 | 50 | 49 | 50 | 49 | 50 | 29 |
| Test | 637 | 80 | 100 | 99 | 100 | 99 | 100 | 59 |

**Key findings from the data exploration**
- Mild class imbalance (technology 9.1%, business 12.6%; largest/smallest ratio 1.72), nearly identical across splits. Macro-F1 is the main metric for this reason.
- Articles are long and uneven: mean 437 words, median 334, maximum 2,940. 57 training articles have fewer than 20 words.
- Training set: about 1.0 million tokens, 30,402 distinct words, 45.5% of them appearing only once. Out-of-vocabulary rate on unseen text: 2.0% (dev) and 2.4% (test).
- Leakage checks: no duplicate rows in train, no URL overlap between splits, and one identical body text shared by train and test.

**Preprocessing.** Input text is `headline + ". " + body`; missing values become empty strings. For the sparse and RNN models, text is lowercased and tokenised with a regex that keeps Hausa hooked letters (ɓ, ɗ, ƙ, ƴ) and apostrophes inside words. There is no stemming or stop-word removal. Transformers use their own subword tokenisers. Long articles are truncated (62% of training articles exceed 256 word tokens).

| Class distribution | Article length |
|---|---|
| ![Class distribution](figures/fig_class_distribution.png) | ![Length distribution](figures/fig_length_distribution.png) |

## Methods

**Evaluation rule.** Every model is trained on the training set and selected by **validation (dev) macro-F1**. The test set is used only to report final numbers. Random seed 42 throughout; each configuration was run once.

**Metrics.** Accuracy, macro-precision, macro-recall, macro-F1 and weighted-F1, plus per-class precision/recall/F1, confusion matrices and an exact McNemar test.

**Sparse models (S1–S7).** TF-IDF (sublinear term frequency, minimum document frequency 2) with Logistic Regression, LinearSVC or Multinomial Naive Bayes; word 1–2 grams or character 2–5 grams; plus headline-only and body-only variants. Class-balanced weights where supported.

**Recurrent models (N1–N4).** Embedding (100-d) → bidirectional LSTM or GRU (128 hidden units per direction) → masked max-pooling or additive attention pooling → dropout 0.4 → linear layer → softmax. Inputs are cut or padded to 256 tokens; the vocabulary (16,571 words with frequency ≥ 2) comes from the training set only. Word2Vec (skip-gram, 100-d, window 5, 30 epochs) is also trained on training text only. Adam (lr 1e-3), batch size 32, gradient clipping 1.0, class-weighted cross-entropy, early stopping on dev macro-F1 (patience 5, at most 25 epochs).

**Transformers (T1–T7).** A pretrained encoder plus a classification head, fine-tuned end to end with a hand-written PyTorch training loop: AdamW (weight decay 0.01), 10% linear warm-up then linear decay, fp16 mixed precision, gradient clipping, class-weighted cross-entropy, 5 epochs (8 for the frozen-encoder run), best epoch kept by dev macro-F1. Batch size 16 (8 for the 512-token run).

| Pretrained model | Hugging Face ID |
|---|---|
| AfroXLMR-base | `Davlan/afro-xlmr-base` |
| AfriBERTa-base | `castorini/afriberta_base` |
| XLM-R-base (generic) | `xlm-roberta-base` |

## All experiments

Full numbers (including dev scores, precision, recall and training times) are in [`experiment_results.csv`](experiment_results.csv). Learning curves are in [`figures/fig_learning_curves.png`](figures/fig_learning_curves.png).

| ID | Approach | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **S1** | **TF-IDF word (1–2) + LogReg C=1 (baseline)** | 0.881 | 0.871 |
| S2 | TF-IDF + LogReg C=10 | 0.893 | 0.885 |
| S3 | TF-IDF + LinearSVC | 0.895 | 0.884 |
| S4 | TF-IDF + Multinomial NB | 0.856 | 0.828 |
| S5 | TF-IDF char n-grams (2–5) + LogReg C=10 | 0.892 | 0.883 |
| S6 | Headline only, TF-IDF + LogReg | 0.799 | 0.790 |
| S7 | Body only, TF-IDF + LogReg | 0.882 | 0.872 |
| N1 | BiLSTM, random embeddings, max-pool | 0.810 | 0.800 |
| N2 | BiLSTM + Word2Vec, max-pool | 0.896 | 0.888 |
| N3 | BiGRU + Word2Vec, max-pool | 0.885 | 0.871 |
| N4 | BiGRU + Word2Vec, attention pooling | 0.896 | 0.884 |
| T1 | XLM-R-base (generic), len 256, lr 2e-5 | 0.876 | 0.867 |
| T2 | AfriBERTa-base, len 256, lr 5e-5 | 0.915 | 0.910 |
| T3 | AfroXLMR-base, len 128, lr 2e-5 | 0.912 | 0.905 |
| T4 | AfroXLMR-base, len 256, lr 2e-5 | 0.911 | 0.903 |
| **T5** | **AfroXLMR-base, len 512, lr 2e-5, bs 8 (selected)** | **0.914** | **0.908** |
| T6 | AfroXLMR-base, len 256, lr 5e-5 | 0.912 | 0.906 |
| T7 | AfroXLMR-base, frozen encoder, len 256, lr 1e-3 | 0.821 | 0.808 |

**What the experiments show**
- The baseline was deliberately untuned. Raising C from 1 to 10 alone gains 1.4 points (S1 → S2), so part of the Transformer's advantage is really tuning.
- LogReg ≈ LinearSVC; Naive Bayes is clearly worse; character n-grams do not beat word n-grams.
- Headline alone (0.790) is much weaker than body alone (0.872); using both is best.
- For RNNs, Word2Vec matters a lot (0.800 → 0.888). Attention pooling helped the BiGRU and trained it faster, but differences between RNN variants are small and based on single runs.
- A tuned TF-IDF model is competitive with the RNNs. More complex did not automatically mean better.

## Error analysis

The selected model makes **55 errors on 637 test articles (8.6%)**. The misclassified articles are saved in [`misclassified_test_examples.csv`](misclassified_test_examples.csv).

- **Most common confusions:** business → politics (6), entertainment → technology (6), business → religion (4), technology → business (4).
- **Successes:** sports is almost perfect (F1 0.98); health (0.95) and religion (0.93) are strong. Sports match reports are classified with near-100% confidence.
- **Why it fails:** (1) genuine topic overlap, e.g. Hajj-related financial stories (business vs religion) and multi-topic weekly round-ups; (2) debatable labels, e.g. a Madonna COVID story labelled entertainment but predicted health; (3) politics and religion overlapping in Nigerian public life; (4) very short articles and spelling variation (Word2Vec learned *daya* and *ɗaya* as near neighbours).
- **Overconfidence:** all eight most confident mistakes had a predicted probability of about 0.996 or higher. The model's probabilities are not calibrated.
- **By article length (accuracy, baseline / T5):** under 50 words 0.818 / 0.909; 50–150 words 0.911 / 0.911; 150–400 words 0.907 / 0.931; over 400 words 0.861 / 0.902.

![Confusion matrices, baseline vs T5](figures/fig_confusion_matrices.png)

## Repository structure

> Adjust this list to match your repository.

```
.
├── README.md
├── hausa_news_classification.ipynb   # full pipeline: EDA, all 18 experiments, evaluation, error analysis, saves model and app files
├── run_app_colab.ipynb               # launches the Gradio app in Google Colab with a public link
├── app.py                            # Gradio app for the fine-tuned model
├── requirements.txt                  # dependencies for the app
├── environment.json                  # software and hardware versions used in my runs
├── experiment_results.csv            # all experiment metrics
├── misclassified_test_examples.csv   # the 55 test errors of the selected model
└── figures/
    ├── fig_class_distribution.png
    ├── fig_length_distribution.png
    ├── fig_experiment_comparison.png
    ├── fig_learning_curves.png
    ├── fig_confusion_matrices.png
    └── fig_per_class_f1.png
```

The data files and the fine-tuned model weights are not stored in this repository (the weights are about 1 GB). Data comes from MasakhaNEWS; the model is on the Hugging Face Hub (link above).

## Reproduce the experiments

All experiments run in one notebook, `hausa_news_classification.ipynb`, on Google Colab.

1. Get the four Hausa files `train.tsv`, `dev.tsv`, `test.tsv` and `labels.txt` from the MasakhaNEWS release (see [Dataset](#dataset)).
2. Upload `hausa_news_classification.ipynb` to Google Colab.
3. Choose **Runtime → Change runtime type → T4 GPU**. The Transformer experiments are skipped without a GPU.
4. Choose **Runtime → Run all**. When the upload box appears, select the four data files. If Colab asks you to restart the runtime after the first install cell, do so, then run all again.
5. The notebook trains and evaluates every experiment, writes the results and figures to a `results` folder, saves the best fine-tuned Transformer to `best_model`, generates `app.py` and `requirements.txt`, and packages the outputs as `results.zip`.

Notes:
- The experiment training times in `experiment_results.csv` add up to roughly 30 minutes on a T4 (plus about 3 minutes for Word2Vec), so allow about 30–45 minutes for a full run.
- Models are selected by dev macro-F1 and the test set is only reported, never used for tuning.
- Random seed: 42. Results can differ slightly between runs because of GPU non-determinism and library versions.
- `SAVE_MODEL_ZIP` (default `False`) in the notebook's configuration cell can be set to `True` to also download the roughly 1 GB model.
- The last cell of the notebook uploads `best_model` to the Hugging Face Hub. Log in and change the repository name to your own account before running it.

**Environment used** (from `environment.json`): Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.18.0, scikit-learn 1.6.1, Tesla T4 GPU, seed 42.

## Run the app

The app takes a Hausa headline and article text, joins them as `headline. text`, tokenises with the model's tokenizer (maximum 512 tokens), and shows the probability of each of the seven topics.

**Option A: Google Colab (public link)**
1. Upload `run_app_colab.ipynb` to Google Colab. A GPU is optional but makes predictions faster.
2. Run the cells in order. If Colab asks you to restart after the install cell, do so and continue from the second cell.
3. Open the `https://xxxx.gradio.live` link printed by the last cell. Keep the notebook tab open; the link stops working when the notebook stops and expires after a limited time.

**Option B: On your own computer**
```bash
git clone [ADD YOUR REPO URL]
cd [your-repo-name]
pip install -r requirements.txt
```
Then, in `app.py`, set `MODEL_ID` to the Hub model (`"dmahe09/hausa-news-classifier"`, or your own repo id) instead of the local `best_model` folder, and run:
```bash
python app.py
```
Open the local address shown in the terminal (usually http://127.0.0.1:7860). The first run downloads the model (about 1 GB), and predictions run on CPU unless you have a GPU.

## Limitations

- Each experiment was run once with one seed, so small differences (for example LSTM vs GRU, or T2 vs T5) are within noise.
- The test set has 637 articles, so one article changes accuracy by about 0.16 points.
- All data is professional news; performance on social media or informal Hausa is unknown.
- Labels are single-topic, but some articles genuinely belong to several topics.
- The baseline's religion vocabulary is dominated by Islamic terms, so the model may be weaker on other religious content (not tested).
- Even at 512 tokens, about half of the training articles are truncated.
- Predicted probabilities are overconfident.
- AfriBERTa and AfroXLMR differ in learning rate as well as in model, so that comparison is not fully controlled.

**Ideas for future work:** multiple seeds with mean ± standard deviation, probability calibration, chunking or hierarchical models for long articles, re-annotating ambiguous articles, multi-label classification, and testing on other Hausa domains.

## Resources used and what I built myself

| Existing resources used | What I implemented |
|---|---|
| MasakhaNEWS dataset; pretrained AfroXLMR, AfriBERTa and XLM-R; scikit-learn, PyTorch, Hugging Face Transformers, gensim, Gradio | Data exploration and leakage checks; all 18 experiment configurations; the BiLSTM/BiGRU models with masked max and attention pooling; the hand-written Transformer fine-tuning loop; evaluation, McNemar test and error analysis; the Gradio app and Colab launcher |

**AI assistance:** [Edit this line to describe any AI tools used and for what, and state that you reviewed and tested the results. Delete this note if not applicable.]

## Licences and acknowledgements

- Dataset: MasakhaNEWS, AFL-3.0. Thanks to the Masakhane community and the dataset authors.
- Pretrained models: AfroXLMR (`Davlan/afro-xlmr-base`), AfriBERTa (`castorini/afriberta_base`) and XLM-R (`xlm-roberta-base`) from the Hugging Face Hub; please check each model card for its licence.
- Code in this repository: [ADD LICENCE, e.g. MIT, or remove this line].

## References

- Adelani, D. I., et al. (2023). MasakhaNEWS: News topic classification for African languages. *Proceedings of IJCNLP-AACL 2023*. arXiv:2304.09972.
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *Proceedings of COLING 2022*.
- Ogueji, K., Zhu, Y., & Lin, J. (2021). Small data? No problem! Exploring the viability of pretrained multilingual language models for low-resourced languages. *Proceedings of the 1st Workshop on Multilingual Representation Learning*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *Proceedings of ACL 2020*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient estimation of word representations in vector space*. arXiv:1301.3781.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP 2014*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP 2020: System Demonstrations*.
- Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *LREC 2010 Workshop on New Challenges for NLP Frameworks*.
- Abid, A., et al. (2019). *Gradio: Hassle-free sharing and testing of ML models in the wild*. arXiv:1906.02569.

The full reference list is in the project report.
