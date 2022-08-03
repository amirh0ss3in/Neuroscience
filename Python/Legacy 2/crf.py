import pandas as pd
import re
import os

def classification_report_df(report):
    report_data = []
    lines = report.split('\n')
    for i, line in enumerate(lines[2:]):
        row = {}
        row_data = re.split(r'\s+', line)
        if len(row_data) > 1 and i<len(lines)-6:
            row['class'] = row_data[1]
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1-score'] = float(row_data[4])
            row['support'] = int(row_data[5])
            report_data.append(row)
        # print(report_data)
        if i == len(lines)-6:
            row['accuracy'] = float(row_data[2]) 
            # print(row_data)
            row['support'] = int(row_data[3])

            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)

    return dataframe

def main(subj_indx):
    ids = [1,3,4,6,7,8,9,10,12,17,18]
    subj_id = ids[subj_indx]
    cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

    subj_path = cwd+ r"Subject wise\\CNN_1D_1_subj"
    subj_path += f"{subj_id}\\"
    k = len(os.listdir(subj_path))
    dfs = []
    for i in range(1,k+1):
        path = subj_path + f"fold{i}\classification_report_fold{i}.txt"
        df = classification_report_df(open(path).read())
        dfs.append(df)


    df_concat = pd.concat((dfs[0], dfs[1], dfs[2], dfs[3], dfs[4]))
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_stds = by_row_index.std()
    support = dfs[0]['support']
    df_means = df_means.round(4)
    df_stds = df_stds.round(4)
    df_means = df_means.astype(str) + ' +/- ' + df_stds.astype(str)
    df_means['support'] = support
    print(df_means)
    df_means.to_csv(subj_path + f"subj{subj_id}_report.csv")

# if __name__ == "__main__":
#     main(0)