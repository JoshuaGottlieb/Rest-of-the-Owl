import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(log_file, column_list, scale = False, train_size = 1905, plot = False):
    '''
    Parses epoch loss logs, returning the results as a formatted Pandas DataFrame.
    
    log_file: str, denoting the file path of the logs.
    column_list: list of str, denoting the names of the columns to use for formatting the Pandas DataFrame.
    scale: bool, denotes whether the scale the non-epoch columns. The models were originally trained with
           a training loop that failed to scale the losses by the training size. Default False.
    train_size: int, denotes the size of the training set. Used if scale is True. Default 1905.
    plot: bool, whether to plot the losses. Default False.
    '''
    
    # Read in logs as a list of strings, with each entry representing an epoch.
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    # Split along field type, then use regex to extract only numerical portions.
    logs_parsed = [x.split(',') for x in logs]
    logs_parsed_numerical = [[re.search(r' (\d+\.?(\d+)?)', y)[1] for y in x] for x in logs_parsed]
    
    # Create a dataframe using the previously parsed data, format column types.
    df = pd.DataFrame(logs_parsed_numerical, columns = column_list)
    non_epoch_columns = [x for x in column_list if x != 'epoch']
    df = df.astype(float)
    df.epoch = df.epoch.astype(int).values
    
    # Due to an error in the original training loop, all loss values were unscaled by training size.
    # If necessary, scale the non-epoch columns.
    if scale:
        df[non_epoch_columns] = df[non_epoch_columns].apply(lambda x: x / train_size)
    
    # Plot the losses.
    if plot:
        fig, ax = plt.subplots(len(non_epoch_columns), 1, figsize = (15, 25))

        plt.tight_layout(pad = 4.0)
        
        for i in range(len(non_epoch_columns)):
            sns.lineplot(x = df.epoch, y = df[non_epoch_columns[i]], ax = ax[i]);
            ax[i].set_xlabel(f'{non_epoch_columns[i]}', fontsize = 16)
            
        plt.xlim([0, 200])
            
    return df

def parse_fid_logs(fid_file, plot = False):
    '''
    Parses FID log files, returning the results as a formatted Pandas DataFrame.     
    
    fid_file: str, denoting the location of the log file containing FID scores.
    plot: bool, whether to plot the FID scores. Default False.
    '''
    
    # Read logs - formatted as one long string.
    with open(fid_file, 'r') as f:
        fids = f.read()
    
    # Use regex to parse the epoch separations, then split along epoch.
    fids_parsed = re.sub(r'(\d)(E)', '\g<1>\n\g<2>', fids).split('\n')
    
    # Split along field type, then use regex to extract only numerical portions.
    fids_cols = [x.split(',') for x in fids_parsed]
    fids_parsed_numerical = [[re.search(r' (\d+\.?(\d+)?)', y)[1] for y in x] for x in fids_cols]
    
    # Create a dataframe using the previously parsed data, format column types.
    df = pd.DataFrame(fids_parsed_numerical, columns = ['epoch', 'fid'])
    df = df.astype(float)
    df.epoch = df.epoch.astype(int).values
    
    # Plot the FID scores.
    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (16, 8))      
        sns.lineplot(x = df.epoch, y = df.fid, ax = ax);
        ax.set_xlabel(f'FID Score', fontsize = 16)
            
        plt.xlim([0, 200])
            
    return df