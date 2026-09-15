# NLP Learning Notes — Day 01
### Foundations: Why NLP is hard, and how text becomes numbers

---

## 0. Where NLP fits (from first principles)

You already know ML: a model is a function `f(X) -> y`, learned from data, where `X` is a matrix of numbers. That's the whole game.

The entire problem of NLP boils down to **one question**: *how do you turn language into `X` without destroying the meaning?*

Images are already numbers (pixel intensities). Tabular data is already numbers. Language is not — it's a sequence of discrete symbols (words) with:
- **Variable length** (a sentence isn't a fixed-size vector like an image is)
- **Order-dependence** ("dog bites man" ≠ "man bites dog")
- **Ambiguity** ("bank" = river bank or money bank?)
- **Sparsity** (vocabulary can be 100,000+ words, but each sentence uses maybe 10-20)

Every NLP technique you'll learn this week is just a different answer to: *"how do I encode variable-length, ambiguous, ordered symbols into fixed-size numeric vectors a model can use?"* Keep asking yourself this question — it's the thread connecting everything.

---

## 1. The NLP Pipeline (the big picture)

```
Raw Text → Preprocessing → Tokenization → Text Representation → Model → Output
```

Today (Day 1) we go deep on the first three steps and touch the beginning of representation. Days 2-5 build the rest.

---

## 2. Preprocessing — cleaning text from first principles

**Why preprocess at all?** A model can't reason about spelling, casing, or punctuation the way humans do — it only sees whatever symbols you hand it. If "Dog", "dog", and "dog." are three different tokens to your model, you've tripled your vocabulary for no semantic gain. Preprocessing exists purely to **reduce noise while preserving meaning**.

### 2.1 Lowercasing
`"Apple" → "apple"`
- **Reasoning:** Case usually doesn't change meaning ("The" vs "the"). Reduces vocabulary size.
- **When NOT to do it:** Named Entity Recognition (NER), sentiment tasks where "AMAZING" (shouting/emphasis) differs from "amazing". Case can be signal, not noise — decide per task.

### 2.2 Removing punctuation / special characters
`"Hello, world!!" → "Hello world"`
- **Reasoning:** Punctuation rarely carries standalone meaning for classic tasks (spam detection, topic classification).
- **When NOT to do it:** Sentiment analysis (exclamation marks/emojis carry emotion), code/URL processing.

### 2.3 Removing stopwords
Stopwords = high-frequency, low-information words: `is, the, a, an, in, of, and...`
- **First-principles reasoning:** In information retrieval, a word's importance is roughly inversely proportional to how often it appears across documents (this idea reappears later as **TF-IDF**). "the" appears everywhere → carries almost no distinguishing signal.
- **When NOT to remove them:** Machine translation, text generation, question answering — stopwords carry grammatical structure needed for fluent output. Removing "not" from "not good" is a classic mistake that flips sentiment meaning.

### 2.4 Stemming vs Lemmatization
Both aim to collapse different forms of a word to one root, so "running", "runs", "ran" don't count as 3 separate vocabulary items.

| | Stemming | Lemmatization |
|---|---|---|
| Method | Chops suffixes using crude rules | Uses vocabulary + grammar (POS tagging) to find the real dictionary root |
| Example | "studies" → "studi" | "studies" → "study" |
| Speed | Fast | Slower |
| Accuracy | Can produce non-words | Always a real word |
| Common algorithm | Porter Stemmer | WordNet Lemmatizer |

**First-principles intuition:** Stemming is a hack (regex-like suffix stripping) — fast but crude. Lemmatization actually understands morphology (needs to know if "meeting" is a noun or verb, since the lemma differs). Use stemming when speed matters more than precision (search engines); use lemmatization when meaning matters more (chatbots, summarization).

---

## 3. Tokenization — the most fundamental step

**Definition:** Splitting text into units (tokens) — usually words, sometimes subwords or characters.

`"I love NLP!" → ["I", "love", "NLP", "!"]`

**Why is this non-trivial?**
- `"don't"` → is this one token or two (`do`, `n't`)?
- `"New York"` → two tokens, but conceptually one entity.
- Languages like Chinese/Japanese have **no spaces** between words at all — tokenization requires a dictionary or a trained model.

**Three levels of tokenization:**
1. **Word-level** — simplest, but vocabulary explodes with rare words ("unhappiness" is a whole new token separate from "happy").
2. **Character-level** — tiny vocabulary (26 letters), but loses word-level meaning, and sequences get very long.
3. **Subword-level** (e.g. Byte-Pair Encoding, WordPiece) — the modern default (used in BERT, GPT). `"unhappiness" → ["un", "happi", "ness"]`. This is the sweet spot: handles rare/unseen words gracefully while keeping sequences short. **You'll meet this again in Deep Learning week** — it's foundational to how transformers process text.

---

## 4. Text Representation — turning tokens into numbers

This is the heart of Day 1. Three techniques, each solving a limitation of the previous one.

### 4.1 One-Hot Encoding (the naive baseline)
Each word = a vector of size `|vocabulary|`, with a single `1` at that word's index.

`vocab = [cat, dog, sat]`
`"cat" = [1, 0, 0]`, `"dog" = [0, 1, 0]`

**Problem (first principles):** Every word is equidistant from every other word (cosine similarity = 0 between any two words). "cat" and "dog" are just as "different" as "cat" and "airplane" — the representation carries **zero semantic information**, and vectors are huge and sparse for large vocabularies.

### 4.2 Bag of Words (BoW)
Represent a whole **document** as a vector of word counts, ignoring order and grammar.

Sentence: `"the cat sat on the cat mat"`
Vocabulary: `[the, cat, sat, on, mat]`
BoW vector: `[2, 2, 1, 1, 1]`

**Reasoning:** If two documents share many of the same words, they're probably about the same topic — order doesn't matter much for topic classification. This is a reasonable first approximation.

**Limitation:** "not good" and "good not" produce the same vector as would any scrambled version — **all word order is destroyed**. Also, every word is weighted equally, so common words dominate the vector even if they're uninformative (this is exactly what stopword removal partially fixes, but doesn't fully solve).

### 4.3 TF-IDF (Term Frequency – Inverse Document Frequency)
This fixes BoW's "all words weighted equally" problem, from first principles:

- **TF (Term Frequency):** how often a word appears in *this* document. High TF → word seems important to this document.
- **IDF (Inverse Document Frequency):** how rare the word is *across all documents*. A word appearing in every document (like "the") gives almost no information about what makes this document distinct.

`TF-IDF(word, doc) = TF(word, doc) × IDF(word)`
`IDF(word) = log( total_documents / documents_containing_word )`

**Intuition:** A word is *important to a document* if it appears often *there* but rarely *elsewhere*. This single idea — "frequent locally, rare globally = high signal" — is one of the most reused principles in all of information retrieval (it's the ancestor of how search engines rank pages, and even echoes in how attention mechanisms in transformers decide what to focus on).

**Still limited by:** No word order, no semantics (still doesn't know "good" and "great" are similar) — this is the exact gap that **word embeddings** (Day 2-3) exist to close.

---

## 5. Hands-on: minimal working code (Python)

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

text = "The cats are running faster than the dogs!"

# 1. Lowercase + tokenize
tokens = nltk.word_tokenize(text.lower())
print("Tokens:", tokens)

# 2. Remove stopwords + punctuation
stop_words = set(stopwords.words('english'))
clean_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
print("Clean tokens:", clean_tokens)

# 3. Stemming vs Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
print("Stemmed:", [stemmer.stem(t) for t in clean_tokens])
print("Lemmatized:", [lemmatizer.lemmatize(t) for t in clean_tokens])

# 4. Bag of Words
corpus = ["the cat sat on the mat", "the dog sat on the log"]
bow = CountVectorizer()
print("BoW:\n", bow.fit_transform(corpus).toarray())
print("Vocabulary:", bow.get_feature_names_out())

# 5. TF-IDF
tfidf = TfidfVectorizer()
print("TF-IDF:\n", tfidf.fit_transform(corpus).toarray())
```

Run this yourself in Colab today — don't just read it. Try changing the corpus and predicting the output before running.

---

## 6. Day 01 Checklist — make sure you can explain (not just recall):
- [ ] Why language can't be fed to a model directly (variable length, no inherent numeric meaning)
- [ ] Why stopword removal helps some tasks and hurts others
- [ ] The difference between stemming and lemmatization, and when each is preferred
- [ ] Why word-level tokenization breaks down for rare words, and how subword tokenization fixes it
- [ ] Why One-Hot Encoding has zero semantic value
- [ ] Why TF-IDF is "smarter" than plain Bag of Words
- [ ] Run the code above and inspect the actual vectors — don't just trust the theory

---

## 7. Five-Day Roadmap

**Day 1 (today) — Foundations**
Preprocessing, tokenization, BoW, TF-IDF. *(this document)*

**Day 2 — Word Embeddings**
Word2Vec (CBOW & Skip-gram), GloVe. First principles: how do you get a vector where "king - man + woman ≈ queen"? Covers the distributional hypothesis ("a word is defined by the company it keeps").

**Day 3 — Sequence Models & Context**
Why order matters and BoW/TF-IDF can't capture it. RNNs, LSTMs, GRUs from first principles — the vanishing gradient problem, and why we needed something better (sets up Day 4).

**Day 4 — Attention & Transformers**
Why RNNs are slow and forget long-range context. Self-attention from first principles ("every word looks at every other word and decides how much to care about it"). This is the conceptual bridge into modern deep learning NLP (BERT, GPT).

**Day 5 — Practical NLP Tasks + Tooling**
Named Entity Recognition, sentiment analysis, text classification end-to-end using spaCy / Hugging Face `transformers`. Wire together preprocessing → embeddings/transformer → classifier into one real pipeline.

**After Day 5 → Real-world projects / Deep Learning**
Once Day 5 is done you'll have both the classical toolkit (TF-IDF, BoW) and the conceptual grounding for embeddings and transformers — enough to either (a) build a real project (e.g. a sentiment classifier, a resume-job matcher, a chatbot intent classifier — all natural given your MERN + freelancing background), or (b) go deeper into Deep Learning NLP (fine-tuning BERT/GPT-style models), since you'll already understand *why* those architectures exist, not just how to call `.fit()`.

---

**Tomorrow (Day 2):** bring questions from today's checklist that you couldn't answer confidently — we'll start there before moving to embeddings.