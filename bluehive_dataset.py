import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import emoji
import pandas as pd
from tqdm import tqdm
import fasttext


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the tweet cleaner function


def tweet_cleaner(unclean_string):
    
    # Step 1: Convert to lowercase
    tweet = unclean_string.lower()

    #remove first char
    tweet = tweet[1:]

    # Step 2: Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    tweet = emoji.replace_emoji(tweet, replace=lambda chars, data_dict: data_dict['en'])

    # tweet = ' '.join([word for word in tweet.split() if '@' not in word and '_' not in word])
    #substitue _ and @ with space
    tweet = re.sub(r'@', ' ', tweet)
    tweet = re.sub(r'_', ' ', tweet)
    
    # Step 4: Remove punctuation except for forward slash
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Step 5: Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    tweet = ' '.join([word for word in tweet.split() if len(word) < 25])
    # tweet = ' '.join([word for word in tweet.split() if not any(c.isdigit() for c in word)])
    
    # Remove underscores

    return tweet


# Map NLTK POS tags to WordNet POS tags
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Lemmatize the tweet
def lemmatize_tweet(tweet):
    tweet = tweet_cleaner(tweet)
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_tweet.append(word)
        else:
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_tweet)

# Extract mentions from tweets
def extract_mentions(text):
    if pd.isna(text):
        return ''
    mentions = []
    for word in text.split():
        if word.startswith('@'):
            cleaned_mention = word.replace('@', '').replace('_', ' ')
            cleaned_mention = re.sub(r'\W+', '', cleaned_mention).lower()
            mentions.append(cleaned_mention)
    return ' '.join(mentions)

# Load datasets
original_train = open('training_data.xlsx', 'rb')
original_train = pd.read_excel(original_train)
original_train['full_text'] = original_train['full_text'].apply(lambda x: x.encode().decode('unicode_escape').encode('latin1').decode('utf8'))

original_test = open('test_data.xlsx', 'rb')
original_test = pd.read_excel(original_test)
original_test['full_text'] = original_test['full_text'].apply(lambda x: x.encode().decode('unicode_escape').encode('latin1').decode('utf8'))


# Process hashtags and mentions
original_train['hashtags'] = original_train['hashtags'].astype(str).str.lower()
original_test['hashtags'] = original_test['hashtags'].astype(str).str.lower()
original_train['at'] = original_train['full_text'].apply(extract_mentions)
original_test['at'] = original_test['full_text'].apply(extract_mentions)

# Apply tweet cleaner with tqdm
tqdm.pandas()
original_train['cleaned_text'] = original_train['full_text'].progress_apply(tweet_cleaner)
original_test['cleaned_text'] = original_test['full_text'].progress_apply(tweet_cleaner)


original_train[['hashtags', 'at', 'cleaned_text', 'country_user', 'gender_user', 'pol_spec_user']].to_csv('cleaned_train.csv', encoding="utf8", index=False)
original_test[['Id', 'hashtags', 'at', 'cleaned_text', 'country_user', 'gender_user']].to_csv('cleaned_test.csv', encoding="utf8", index=False) 


# Load cleaned data
cleaned_train = pd.read_csv('cleaned_train.csv', encoding='utf8')
cleaned_test = pd.read_csv('cleaned_test.csv', encoding='utf8')

# # Language detection using FastText
model = fasttext.load_model("lid.176.bin")

def detect_language_safe(text):
    if not isinstance(text, str) or not text.strip() or text == "unknown":
        return "unknown"
    try:
        prediction = model.predict(text, k=1)
        return prediction[0][0].replace("__label__", "")
    except Exception:
        return "error"


tqdm.pandas()
cleaned_train['language'] = cleaned_train['cleaned_text'].progress_apply(detect_language_safe)
cleaned_test['language'] = cleaned_test['cleaned_text'].progress_apply(detect_language_safe)

# Save language-detected data
cleaned_train.to_csv('cleaned_train_data_language_augmented.csv', index=False, encoding='utf8')
cleaned_test.to_csv('cleaned_test_data_language_augmented.csv', index=False, encoding='utf8')

# Augment the data with additional metadata
df = pd.read_csv('cleaned_train_data_language_augmented.csv', encoding='utf8')
test_df = pd.read_csv('cleaned_test_data_language_augmented.csv', encoding='utf8')

columns_to_fill = ['hashtags', 'at', 'gender_user', 'country_user', 'language', 'cleaned_text']
df[columns_to_fill] = df[columns_to_fill].fillna('nan')
test_df[columns_to_fill] = test_df[columns_to_fill].fillna('nan')

df['cleaned_text'] = df['hashtags'] + " [SEP] " + df['at'] + " [SEP] " + \
                     df['gender_user'] + " [SEP] " + df['country_user'] + " [SEP] " + \
                     df['language'] + " [SEP] " + df['cleaned_text']

test_df['cleaned_text'] = test_df['hashtags'] + " [SEP] " + test_df['at'] + " [SEP] " + \
                          test_df['gender_user'] + " [SEP] " + test_df['country_user'] + " [SEP] " + \
                          test_df['language'] + " [SEP] " + test_df['cleaned_text']

# # Save final augmented data
df.to_csv('cleaned_train_data_language_augmented.csv', index=False, encoding='utf8')
test_df.to_csv('cleaned_test_data_language_augmented.csv', index=False, encoding='utf8')
