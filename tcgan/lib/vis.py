import matplotlib.pyplot as plt
import os


def save_series(samples, path):
    # just show 4 * 4 samples
    fig = plt.figure(figsize=(16, 8))  # each sample takes an area of 4*2.
    n_samples = min(samples.shape[0], 16)
    for i in range(n_samples):
        plt.subplot(4, 4, i + 1)
        plt.plot(samples[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_loss(df, path):
    df.to_csv(os.path.join(path, 'metric.csv'))
    epochs = df['epoch']
    metric_df = df.drop(columns=['epoch'])
    for col in metric_df:
        plt.plot(epochs, metric_df[col].values)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'metric_{col}.png'))
        plt.close()

