# README for AC 221 Project

## Proposal

See `ac221_project_proposal.pdf`

## Timeline

- `collect.py`: Tweet data collection
- `agg.py`: Aggregate collected partitions into a single csv
- `clean.py`: Clean csv files and add a month column
- `sentiment_transfer.py`
    - Attempted transfer learning on pre-trained hugging face transformer using labeled tweet data (http://help.sentiment140.com/for-students) to categorize sentiment
    - Problem: training takes really long
- `sentiment_pre.py`
    - Just use a pre-trained model from hugging face instead (cardiffnlp/twitter-roberta-base-sentiment)
    - This model is developed and fine-tuned by Cardiff NLP on tweet data specifically for sentiment analysis tasks. The model is based on the RoBERTa architecture, which is an improved version of BERT. Since it has been fine-tuned on tweet data, it performs well on tasks related to tweet text analysis.
- `topic_lda.py`
    - Topic modeling with LDA
- `topic_bert.py`
    - Topic modeling with BERTopic (https://github.com/MaartenGr/BERTopic)
    - BERTopic is a topic modeling technique that utilizes transformer models like BERT, RoBERTa, or DistilBERT to create dense document representations before clustering them to form topics. This approach can handle short text and noisy data better than traditional LDA. You can find more information about BERTopic here.
    - Downside: transformers are really slow; very computationally expensive for big data
        - It takes 45 seconds to process 100 tweets
