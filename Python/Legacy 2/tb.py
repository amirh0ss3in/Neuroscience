import pandas as pd
import requests
from csv import reader
import os


def URLs(fold, trial):
    URLs_dict = {
    'train_accuracy' : f'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_accuracy&run=fold{fold}%5C{trial}%5Cexecution0%5Ctrain&format=csv',
    'val_accuracy' : f'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_accuracy&run=fold{fold}%5C{trial}%5Cexecution0%5Cvalidation&format=csv',
    'val_loss' : f'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_loss&run=fold{fold}%5C{trial}%5Cexecution0%5Cvalidation&format=csv',
    'train_loss' : f'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_loss&run=fold{fold}%5C{trial}%5Cexecution0%5Ctrain&format=csv'
    }
    return URLs_dict

def tb_data(log_dir, mode, fold, num_trials):
    trials = os.listdir(log_dir)
    fdf = {}
    for i, trial in enumerate(trials[:num_trials]):
        r = requests.get(URLs(fold, trial)[mode], allow_redirects=True)
        data = r.text
        data_csv = reader(data.splitlines())
        data_csv = list(data_csv)
        df = pd.DataFrame(data_csv)
        headers = df.iloc[0]
        df  = pd.DataFrame(df.values[1:], columns=headers)
        if i == 0:
            fdf['Step'] = df['Step']  
        fdf[f'trial {trial}'] = df['Value']
    fdf = pd.DataFrame(fdf)
    return fdf

def main():
    ids = [1,3,4,6,7,8,9,10,12,17,18]
    subj_number = 0
    cwd = r'C:/Users/Amirhossein/Desktop/Uni/Neuroscience/Python/'
    fold = 1
    num_trials = 15

    log_dir = cwd+'trial_logs/' + f'CNN_1D_1_subj{ids[subj_number]}/fold{fold}'


    train_accuracy = tb_data(log_dir = log_dir, mode = 'train_accuracy', fold = fold, num_trials = num_trials)
    val_accuracy = tb_data(log_dir = log_dir, mode = 'val_accuracy', fold = fold, num_trials = num_trials)
    val_loss = tb_data(log_dir = log_dir, mode = 'val_loss', fold = fold, num_trials = num_trials)
    train_loss = tb_data(log_dir = log_dir, mode = 'train_loss', fold = fold, num_trials = num_trials)

    train_accuracy.to_csv(f'{log_dir}/train_accuracy_subj{ids[subj_number]}_fold{fold}.csv')
    val_accuracy.to_csv(f'{log_dir}/val_accuracy_subj{ids[subj_number]}_fold{fold}.csv')
    val_loss.to_csv(f'{log_dir}/val_loss_subj{ids[subj_number]}_fold{fold}.csv')
    train_loss.to_csv(f'{log_dir}/train_loss_subj{ids[subj_number]}_fold{fold}.csv')

if __name__ == '__main__':
    main()
