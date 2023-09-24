WEB INTELLIGENCE
Refers to use of Artificial Intelligence, Data Mining, Machine Learning, etc.
Includes many techniques: Web Mining, Sentiment Analysis, Predictive Analysis, Fraud Detection, NLP.

FEATURE ENGINEERING
Bag of Words (BoW): Collection of words disregarding the grammer and meaning, for text analysis (NLP).
Term Frequency-Inverse Document Frequency) (TF-IDF: Extension of BoW. Considers importance of words relative to their Frequency.
Word Embedding: Dense Vector Representation of words that capture semantic relationship between words (Word2Vec, FastText)
Named Entity Recognition (NER): Identifying and classifying names of person, company, locations, dates, etc.
Part-of-Speech Tagging(POS): Labeling each word with its grammatical category (Noun, preposition, verb, etc)
POS continued: Important for various NLP Tasks like text parsing, information extraction, sentiment analysis, etc.
POS continued: Hidden Markov Model, Maximum Entrovpy Markov model, LSTM, are used for POS Tagging and achieving high accuracy in NLP apps.
Sentiment Analysis Features: Emotions expressed by a piece of text, Frequency of positive/negetive words/sentiment score/emotion analysis
Text Length and Structure Features: Text Length, paragraphs count, headings count and presence HTML tags useful in content categorization Readability Features: Readability of text using Flesch Kincaid Grade Level, Gunning Fog Index used for content delivery
Domain Specific Features: Uniqueness of the content can be used for feature engineering.

Unigram: Single word or token
N-gram: N words grouped together. Help in capturing local context and relationship between words. Text Generation, Speech Recognition, etc.
Collocation: Linguistic Phenomena where certain words often occur together (natural or idiomatic in a language), language specific.


Levenschtein Distance: Calculates distance between two words to check for smimilarity. Does this by insertion, deletion, replacement.
LD Continued: Create initial matrix, same char-fill the diagonal number, else fill the +1 of the least number from the 3 neighbours.

Kullback Leibler Divergence: Quantify the difference between two probabilty distribution.
DKL(P||Q) = (summation)[P(i)*log(P(i)/Q(i))], where P and Q are probability distribution and i is the event.
DKL(P||Q) = 0 when P and Q are identical
Used in informatioon retrieval, Text Generation, Reinforcement Learning

t-test: (u1 - u2) / s*root(1/na + 1/nb) and s(pooled standard deviation) = root[(var(a) + var(b)) / (n1 + nb - 2)]


One hot encoding: Classify into binary numbers
Label Encoding: Give label to each category. Straightforward, tree based algo | Priority Issue, Not suitable for feature space, 
Count or Frequency Encoding: (Total numbers of a category / total). Pros and cons same as above
Ordered Label Encoding: More the mean, lower will be its assigned number. Target mean = (Number of males in a category) / (Total count of a category)














