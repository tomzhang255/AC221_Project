# Purpose: fine-tune a pre-trained transformer for sentiment analysis on tweet data
# Need a labeled dataset: http://help.sentiment140.com/for-students

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


if __name__ == '__main__':

    # load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3)

    # got labeled sentiment analysis tweet data from:
    # http://help.sentiment140.com/for-students
    # combine train and test (we will split later)
    labeled_train = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', header=None, encoding='ISO-8859-1')
    labeled_test = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', header=None, encoding='ISO-8859-1')

    labeled_data = pd.concat([labeled_train, labeled_test], axis=0)
    headers = ['label', 'id', 'date', 'query', 'user', 'text']
    labeled_data.columns = headers
    labeled_data.to_csv('trainingandtestdata/labeled_data.csv', index=False)
    labeled_data = pd.read_csv('trainingandtestdata/labeled_data.csv', usecols=['label', 'text'])

    # load labeled dataset
    df = pd.read_csv("trainingandtestdata/labeled_data.csv")
    # use a stratified sample
    # df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=100))

    # labels = 0, 2, 4 - change it to 0, 1, 2
    df['label'] = df['label'].map({0: 0, 2: 1, 4: 2})

    # split dataset into training, validation and testing sets
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # tokenize text data
    def tokenize_text(text):
        return tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='tf')

    train_tokenized_data = tokenize_text(train_df['text'].tolist())
    val_tokenized_data = tokenize_text(val_df['text'].tolist())
    test_tokenized_data = tokenize_text(test_df['text'].tolist())

    # convert sentiment scores to one-hot vectors
    num_classes = 3
    train_sentiment_labels = tf.keras.utils.to_categorical(
        train_df['label'], num_classes=num_classes)
    val_sentiment_labels = tf.keras.utils.to_categorical(
        val_df['label'], num_classes=num_classes)
    test_sentiment_labels = tf.keras.utils.to_categorical(
        test_df['label'], num_classes=num_classes)

    # extract tensors from tokenized_data
    train_input_ids = train_tokenized_data['input_ids']
    train_attention_mask = train_tokenized_data['attention_mask']

    val_input_ids = val_tokenized_data['input_ids']
    val_attention_mask = val_tokenized_data['attention_mask']

    test_input_ids = test_tokenized_data['input_ids']
    test_attention_mask = test_tokenized_data['attention_mask']

    # define model training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # train the model on the training dataset with validation
    model.fit({'input_ids': train_input_ids, 'attention_mask': train_attention_mask},
              train_sentiment_labels, epochs=10, batch_size=32,
              validation_data=({'input_ids': val_input_ids, 'attention_mask': val_attention_mask}, val_sentiment_labels))

    # save model
    model.save('sentiment')

    # load the saved model
    loaded_model = tf.keras.models.load_model('sentiment')

    # evaluate the model on the testing dataset
    test_loss, test_accuracy = model.evaluate(
        {'input_ids': test_input_ids, 'attention_mask': test_attention_mask}, test_sentiment_labels)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
