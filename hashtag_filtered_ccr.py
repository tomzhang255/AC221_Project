# purpose:          apply CCR on hashtag-filtered tweets
# questionnaires:
#   The Generalized Anxiety Disorder 7 (GAD-7):
#       The GAD-7 is a seven-item questionnaire that is used to screen for anxiety.
#   The Beck Depression Inventory (BDI):
#       The BDI is a 21-item questionnaire that is used to assess the severity of depression.
# transformer used: multi-qa-MiniLM-L6-cos-v1;
#                   it's the fastest for sentence embeddings
#                   and supports a max sequence length of 512 (words)
# info on models:   https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/

import warnings
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from datetime import datetime
t1 = datetime.now()
t2 = datetime.now()
print('====================')
print(f'Took {(t2 - t1).total_seconds()} seconds to load sentence_transformers')
warnings.filterwarnings("ignore")


def log(msg):
    print(f'{datetime.now()}; {msg}')


def encode_column(model, df, col_name):
    df["embedding"] = list(model.encode(df[col_name]))
    return df


def item_level_ccr(data_encoded_df, questionnaire_encoded_df):
    q_embeddings = questionnaire_encoded_df.embedding
    d_embeddings = data_encoded_df.embedding
    similarities = util.pytorch_cos_sim(d_embeddings, q_embeddings)
    for i in range(1, len(questionnaire_encoded_df)+1):
        data_encoded_df["sim_item_{}".format(i)] = similarities[:, i-1]
    return data_encoded_df


def ccr_wrapper(data_df, data_col, q_df, q_col):
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    q_encoded_df = encode_column(model, q_df, q_col)
    data_encoded_df = encode_column(model, data_df, data_col)

    ccr_df = item_level_ccr(data_encoded_df, q_encoded_df)
    ccr_df = ccr_df.drop(columns=["embedding"])

    return ccr_df


if __name__ == '__main__':
    df_d = pd.read_csv('data/hashtag_filtered_tweets.csv')

    # BDI
    df_bdi = pd.read_csv('ccr/bdi.csv')
    # apply CCR
    res_bdi = ccr_wrapper(df_d, 'text', df_bdi, 'q')
    # get cosine similarie scores
    sim_scores_bdi = res_bdi.loc[:, res_bdi.columns.str.startswith('sim_item')]
    # average scores
    sim_scores_bdi_mean = pd.DataFrame(
        {'bdi_mean': sim_scores_bdi.mean(axis=1)})

    # GAD-7
    df_gad = pd.read_csv('ccr/gad-7.csv')
    res_gad = ccr_wrapper(df_d, 'text', df_gad, 'q')
    sim_scores_gad = res_gad.loc[:, res_gad.columns.str.startswith('sim_item')]
    sim_scores_gad_mean = pd.DataFrame(
        {'gad_mean': sim_scores_gad.mean(axis=1)})

    # concat both questionnaire results to original df
    res = pd.concat([pd.read_csv('hashtag_filtered_tweets.csv'),
                     sim_scores_bdi_mean, sim_scores_gad_mean], axis=1)
    res.to_csv('data/hashtag_filtered_ccr.csv', index=False)
