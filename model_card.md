# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
Both models were implemented and compared:
- **Rule-based model** (`mood_analyzer.py`) — hand-crafted scoring rules over a word lexicon
- **ML model** (`ml_experiments.py`) — bag-of-words CountVectorizer + Logistic Regression trained on `SAMPLE_POSTS`

**Intended purpose:**
Classify short text messages (social media posts, chat messages) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How the rule-based model works:**
Each input sentence is tokenized and scored word-by-word. Positive words add +1 to the score; negative words subtract 1. Booster words (`so`, `very`, `really`, `wicked`) double the next word's contribution. Negation (`not`) within a 2-token window flips the sign of a positive word or neutralizes a negative one. Emojis are mapped to keyword tokens before scoring. If the final score is positive the label is `positive`; negative → `negative`; zero with both positive and negative signals → `mixed`; zero with neither → `neutral`. Explicit sarcasm markers (`sarcastically`, `#sarcasm`) flip the final score.

**How the ML model works:**
Each post is converted into a bag-of-words vector (word counts) using `CountVectorizer`. A `LogisticRegression` classifier is then trained on those vectors paired with their labels. At prediction time, the same vectorizer transforms the new text and the classifier picks the most likely label based on weights learned during training.

---

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` contains 16 short posts written in the style of social media messages or chat logs. Each post has a matching human-assigned label in `TRUE_LABELS`.

**Labeling process:**
Labels were assigned by reading each post and identifying its dominant feeling. Posts where two feelings were clearly present (e.g., tired but proud) were labeled `mixed`. Posts with no clear sentiment signal were labeled `neutral`.

Hard-to-label examples:
- `"Feeling tired but kind of hopeful"` — tired is negative, hopeful is positive; either `mixed` or `negative` could be argued.
- `"I'm excited for the party but also nervous."` — excitement and anxiety are both present; labeled `mixed` but could be `positive`.
- `"This is lowkey the best pizza I've ever had."` — understated positive; some readers might hear this as sarcastic.

**Important characteristics of the dataset:**

- Contains slang (`lowkey`, `no cap`, `fire`, `sick`)
- Includes emojis (text emoticons and Unicode: 🥲, 😩, 💀)
- Includes sarcasm (`"I absolutely love being ignored. #sarcasm"`)
- Several posts express genuinely mixed feelings
- Some posts are ambiguous even to human readers

**Possible issues with the dataset:**

- Only 16 examples — far too small to represent the full range of human expression
- Labels reflect one person's interpretation; another annotator might disagree on several entries
- The dataset skews toward English internet slang and may not generalize to other dialects or registers
- No explicit test split — accuracy is measured on the same data the word lists were tuned against

---

## 3. How the Rule Based Model Works

**Scoring rules:**

| Rule | Effect |
|------|--------|
| Positive word hit | +1 to score |
| Negative word hit | −1 to score |
| Booster word immediately before (`so`, `very`, `really`, `wicked`) | ×2 the next word's score |
| `not` within 2 tokens before a positive word | Flips score to negative |
| `not` within 2 tokens before a negative word | Neutralizes score to 0 |
| Emoji → keyword mapping (e.g. `😭` → `crying_face_emoji`) | Emoji sentiment enters scoring |
| Hyphenated words split on `-` (e.g. `bitter-sweet` → `bitter` + `sweet`) | Compound words scored independently |
| Sarcasm marker token (`sarcastically`, `sarcasm`) | Final score multiplied by −1 |
| `score > 0` | Label: `positive` |
| `score < 0` | Label: `negative` |
| `score == 0`, both pos + neg signals present | Label: `mixed` |
| `score == 0`, no signals | Label: `neutral` |

**Strengths of this approach:**

- Fully transparent — every prediction can be traced to specific token hits
- Handles slang by extending the word lists (`sick`, `fire`, `meh`)
- Emoji-aware via a mapping table
- Detects mixed mood when opposing signals cancel out
- The 2-token negation window correctly handles phrases like `"not very happy"`

**Weaknesses of this approach:**

- **Cannot detect implicit sarcasm.** "I love getting stuck in traffic" and "Oh great, another Monday" score positive because `love` and `great` are in the word list. Without an explicit marker there is no way to know the speaker is being sarcastic.
- **Vocabulary is the bottleneck.** Any sentiment word not in the lists is silently ignored. "I'm exhausted but proud" only works because `exhausted` and `proud` were manually added.
- **Context-blind.** The model scores tokens independently; "tears of joy" scores `joy` (+1) and `crying` (−1) and calls it mixed, which happens to be acceptable — but for the wrong reason.
- **Unknown slang breaks silently.** "Lowkey the best" scores neutral because `lowkey` is not a booster in the current lists.

---

## 4. How the ML Model Works

**Features used:**
Bag-of-words representation via `CountVectorizer`. Each post becomes a vector of word counts over the training vocabulary. No preprocessing (lowercasing, stop-word removal, or emoji handling) is applied before vectorization.

**Training data:**
The model trained on all 16 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
Because the model trains and evaluates on the same 16 examples, it achieves 100% training accuracy — it has effectively memorized the dataset. This does not reflect real generalization ability.

**Strengths:**
- Learns patterns from data automatically without hand-crafting rules
- Correctly handles posts the rule model misses — e.g., `"Feeling tired but kind of hopeful"` (labeled `mixed`) and `"I'm excited for the party but also nervous."` — because the full word context is captured in the training vectors
- Gets `"This is the best day ever! :D"` right even without emoji support, by learning `best` and `ever` as positive signals

**Weaknesses:**
- **100% accuracy is memorization, not learning.** With only 16 training examples the model has seen every test case during training.
- **No generalization to unseen vocabulary.** On breaker sentences, words like `wicked`, `meh`, `okay`, and `interesting` that never appeared in training default to neutral or get dominated by other words.
- **No preprocessing.** Emojis are silently dropped. `😩` and `🙂` contribute nothing.
- **Brittle on out-of-distribution sentences.** "My vacation was just okay, I guess" was predicted `negative` — likely because `guess` or `just` appeared in negative training examples, not because the model understood hedging.

---

## 5. Evaluation

**How the models were evaluated:**
Both models were run on all 16 labeled posts in `SAMPLE_POSTS` and compared against `TRUE_LABELS`. Both were then run on the 15 breaker sentences (no ground-truth labels — outputs compared against each other). No held-out test set was used.

### SAMPLE_POSTS accuracy

| Model | Correct | Accuracy |
|-------|---------|----------|
| Rule-based | 11 / 16 | 69% |
| ML (Logistic Regression) | 16 / 16 | **100%** ⚠️ training accuracy only |

The ML model's 100% reflects memorization of its training data, not generalization.

### Breaker sentences head-to-head

The two models agreed on 6 of 15 breakers and differed on 9.

| Sentence | Rule | ML | Notes |
|----------|------|----|-------|
| "I love getting stuck in traffic." | positive | **negative** | ML picks up `stuck`/`traffic` as negative context; rule fires on `love` blindly |
| "This party is sick." | positive | **negative** | `sick` is in the rule lexicon as slang-positive; ML never saw it as positive in training |
| "This is wicked awesome." | positive | **negative** | ML has no training signal for `wicked awesome`; likely weights `wicked` as negative |
| "I'm fine 🙂" | positive | **neutral** | Rule model maps `🙂` → `happy_emoji`; ML drops the emoji silently |
| "Oh great, another Monday." | positive | **negative** | ML associates `Monday` / `another` with negativity from training context; rule fires on `great` |
| "My vacation was just okay, I guess." | neutral | **negative** | Rule finds no signal; ML over-generalizes `I guess` / `just okay` as negative |
| "The movie was... interesting." | neutral | **negative** | Both wrong in different ways — neither understands diplomatic understatement |
| "This is the best day ever. (said on a terrible day)" | **negative** | positive | Rule catches `terrible` in the parenthetical; ML focuses on `best day ever` |
| "It's a bitter-sweet moment." | **mixed** | negative | Rule splits hyphen → `bitter`(−1) + `sweet`(+1) = mixed; ML sees `bitter` dominate |

**Examples of correct predictions (rule-based):**

| Post | Predicted | True | Why it worked |
|------|-----------|------|---------------|
| `"I love this class so much"` | positive | positive | `love` in POSITIVE_WORDS; clear single signal |
| `"I absolutely love being ignored. #sarcasm"` | negative | negative | `#sarcasm` → `sarcasm` token triggers score flip |
| `"Just finished a marathon, I'm tired but proud."` | mixed | mixed | `tired`(−1) + `proud`(+1) = 0 with both signals → mixed |

**Examples of incorrect predictions (rule-based):**

| Post | Predicted | True | Why it failed |
|------|-----------|------|---------------|
| `"Feeling tired but kind of hopeful"` | negative | mixed | `hopeful` not in POSITIVE_WORDS; positive half invisible |
| `"This is the best day ever! :D"` | neutral | positive | `best` not in POSITIVE_WORDS; `:D` not in emoji map |
| `"I'm so done with all this work. 😩"` | neutral | negative | `😩` not in emoji map; `done` not in NEGATIVE_WORDS |

---

## 6. Limitations

**Both models:**
- **Tiny dataset.** 16 examples cannot represent the breadth of natural language.
- **No test split.** Accuracy is measured on the same data used to tune both models, so reported numbers overstate real-world performance.
- **English only.** Neither model generalizes to other languages or non-Western cultural expressions of mood.

**Rule-based model:**
- **Implicit sarcasm is undetectable.** "I love getting stuck in traffic" and "Oh great, another Monday" both score positive — there is no way to know the speaker is being sarcastic without external context.
- **Vocabulary is the bottleneck.** Any sentiment word missing from the lists is silently ignored. Gaps found during breaker testing: `hopeful`, `best`, `done`, `okay`, `interesting`.
- **Context-blind beyond a 2-token window.** Sentence structure, topic, and discourse play no role.

**ML model:**
- **100% training accuracy is memorization.** The model has seen every labeled example during training; its accuracy on new text is unknown.
- **No preprocessing.** Emojis, punctuation, and repeated characters are not normalized before vectorization, so `😩` and `🙂` contribute nothing.
- **Out-of-vocabulary brittleness.** Words not seen in training (most of the breaker vocabulary) have zero weight, causing the model to fall back on spurious co-occurrence signals.

---

## 7. Ethical Considerations

- **Mental health risk.** A message expressing genuine distress (e.g., "I'm crying myself to sleep") could be misclassified as positive if the model only catches a positive word. Acting on a wrong mood classification in a supportive context could cause real harm.
- **Cultural and linguistic bias.** The word lists reflect one cultural perspective on sentiment. Words that are neutral in one community (`sick`, `fire`) may mean something completely different in another. Misclassification rates will be higher for dialects and registers not represented in the training data.
- **Privacy.** Analyzing personal messages to infer mood, even with a simple rule-based system, involves processing private information. Users should be informed and consent should be obtained before deploying any mood analysis on real messages.
- **Overconfidence.** The model always returns a label, even when it has detected no signal at all. A `neutral` prediction and a genuinely ambiguous prediction look identical from the outside.

---

## 8. Ideas for Improvement

**Quick wins (both models):**
- **Expand the dataset** to 200–500 labeled examples with a held-out test split so accuracy numbers are meaningful.
- **Add a confidence threshold** — both models should be able to say "I'm not sure" instead of always returning a label.

**Rule-based model:**
- **Expand the emoji map** — `:D`, `😩`, `🙏`, and many others are currently unhandled; these showed up as failures in the breaker test.
- **Extend the lexicon** — `hopeful`, `best`, `done`, `interesting`, `okay` were all missed; a sentiment dictionary (e.g., AFINN or SentiWordNet) would close many of these gaps automatically.
- **Separate the lexicon from the code** — storing word lists in a config file would let non-programmers update them without touching Python.

**ML model:**
- **Add preprocessing** — run the same `MoodAnalyzer.preprocess()` pipeline (emoji mapping, normalization, hyphen splitting) before vectorization so the ML model benefits from the same signal cleaning.
- **Use TF-IDF** instead of raw counts to down-weight extremely common words like `the`, `I`, `is`.
- **Create a real train/test split** — even a 12/4 split would give a more honest accuracy estimate than 100% training accuracy.
- **Use a pre-trained transformer** (e.g., DistilBERT fine-tuned on sentiment data) — transformers handle sarcasm, slang, and context far better than bag-of-words, and pre-training means they generalize without needing hundreds of labeled examples.
