import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualize import configure_axislabels_and_title, configure_ticklabels_and_params, configure_legend
import numpy as np

def parse_logs(log_file, column_list, scale = False, train_size = 1905, plot = False):
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    logs_parsed = [x.split(',') for x in logs]
    logs_parsed_numerical = [[re.search(r' (\d+\.?(\d+)?)', y)[1] for y in x] for x in logs_parsed]
    
    df = pd.DataFrame(logs_parsed_numerical, columns = column_list)
    non_epoch_columns = [x for x in column_list if x != 'epoch']
    df = df.astype(float)
    df.epoch = df.epoch.astype(int).values
    if scale:
        df[non_epoch_columns] = df[non_epoch_columns].apply(lambda x: x / train_size)
    
    if plot:
        fig, ax = plt.subplots(len(non_epoch_columns), 1, figsize = (15, 25))

        plt.tight_layout(pad = 4.0)
        
        for i in range(len(non_epoch_columns)):
            sns.lineplot(x = df.epoch, y = df[non_epoch_columns[i]], ax = ax[i]);
            ax[i].set_xlabel(f'{non_epoch_columns[i]}', fontsize = 16)
            
        plt.xlim([0, 200])
            
    return df

def plot_gen_and_disc_losses(df, model_name):
    fig, ax = plt.subplots(2, 1, figsize = (28, 18))
    sns.lineplot(x = df.epoch, y = df.gen_total_loss, label = 'Generator Total Loss', ax = ax[0]);
    sns.lineplot(x = df.epoch, y = df.disc_loss, label = 'Discriminator Loss', ax = ax[1]);

    configure_axislabels_and_title('Epochs', 'Generator Total Loss',
                                   f'{model_name} Generator Loss Across Epochs', ax = ax[0]);
    configure_axislabels_and_title('Epochs', 'Discriminator Total Loss',
                                   f'{model_name} Discriminator Loss Across Epochs', ax = ax[1]);
    
    ax[1].hlines(np.log(2), 0, 200, colors = 'red', label = 'Ideal Discriminator Loss')
    
    for axis in ax:
        configure_ticklabels_and_params(ax = axis);
        configure_legend(ax = axis, fancybox = True, frameon = True, fontsize = 16);
        
    plt.tight_layout(pad = 3.0)

    return fig, ax