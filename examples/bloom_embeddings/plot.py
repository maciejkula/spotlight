import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import seaborn as sns

from example import Results


def summarise_results(results):

    sns.set_style("darkgrid")

    data = pd.DataFrame([x for x in results])
    data = data[data['embedding_dim'] <= 64]

    best = (data.sort_values('test_mrr', ascending=False)
            .groupby('compression_ratio', as_index=False).first())

    # Normalize per iteration
    best['elapsed'] = best['elapsed'] / best['n_iter']

    print(best)

    baseline_mrr = (best[best['compression_ratio'] == 1.0]
                    ['validation_mrr'].values[0])
    baseline_time = (best[best['compression_ratio'] == 1.0]
                     ['elapsed'].values[0])

    compression_ratio = best['compression_ratio'].values
    mrr = best['validation_mrr'].values / baseline_mrr
    elapsed = best['elapsed'].values / baseline_time

    plt.ylabel("MRR ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs MRR ratio")

    plt.plot(compression_ratio, mrr,
             label='Movielens')
    plt.legend(loc='lower right')
    plt.savefig('plot.png')
    plt.close()

    plt.ylabel("Time ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs time ratio")

    plt.plot(compression_ratio, elapsed)
    plt.savefig('time.png')


if __name__ == '__main__':

    results = Results('results.txt')
    summarise_results(results)
