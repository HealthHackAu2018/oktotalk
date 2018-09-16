from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

data = pd.read_csv('data/psych.csv', encoding='utf-8')
data = data.set_index('psych_id')


def combine(row):
    callouts = ""
    for column in ['headline', 'callout1', 'callout2', 'callout3']:
        if row[column] and not pd.isna(row[column]):
            callouts += "{} ".format(row[column])

    if not len(callouts.strip()):
        return None
    return callouts.strip()


data['callouts'] = data.apply(combine, axis=1)

for text_field in ['description', 'callouts']:
    complete = data.dropna(subset=[text_field])

    tfidf = TfidfVectorizer(max_features=50,
                            sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8',
                            ngram_range=(1, 3),
                            stop_words='english')

    features = tfidf.fit_transform(complete[text_field])

    text_columns = pd.DataFrame(features.A, columns=tfidf.get_feature_names(), index=complete.index)
    text_columns = text_columns.rename(columns=lambda x: f"{text_field}_{x}".replace(" ", "_"))

    text_columns = text_columns.join(complete[[text_field]])

    text_columns.to_csv(f'data/psych_{text_field}.csv')
    print(text_columns.head())
