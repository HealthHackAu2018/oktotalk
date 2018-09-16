# Encode psychologist skills stored as categorial integers into one-hot scheme
# Author terryf82 https://github.com/terryf82

import argparse
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

BASE_FP = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
DATA_FP = os.path.join(BASE_FP, "data")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str,
                        help="psycholgist skills csv")

    args = parser.parse_args()
    
    # load the psychologist skills csv data & convert to pandas dataframe
    df_psych_skills = pd.read_csv(os.path.join(DATA_FP, args.file),
        dtype={"issue_id": "str", "level_id": "str"})
    df_psych_skills.dropna(how="all", inplace=True)
    # df_psych_skills.assign(issue_id_level_id=df_psych_skills.issue_id % "-" % df_psych_skills.level_id)
    # df_psych_skills.assign(issue_id_level_id=df_psych_skills.issue_id)
    df_psych_skills["issue_id_level_id"] = df_psych_skills.apply(lambda row: str(row.issue_id) + "-" + str(row.level_id), axis=1)
    print(df_psych_skills.get_values())
    # print(df_psych_skills)
    
    # encode categories as one hot
    # enc = OneHotEncoder()
    # enc.fit(df_psych_skills)

    # one_hot_df_psych_skills = pd.get_dummies(df_psych_skills, columns=["issue_id", "level_id"])
    # print(one_hot_df_psych_skills)
    
    # psych_skills = df_psych_skills.to_dict()
    # print(psych_skills)
    # print("ready")
    