import pandas as pd

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import seaborn as sns

from example import Results


def summarise_results(results):

    sns.set_style("darkgrid")

    plt.ylabel("MRR ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs MRR ratio")

    data = pd.DataFrame([x for x in results])

    best = (data.sort_values('test_mrr', ascending=False)
            .groupby('compression_ratio', as_index=False).first())

    print(best)

    baseline_mrr = (best[best['compression_ratio'] == 1.0]
                    ['validation_mrr'].values[0])

    compression_ratio = best['compression_ratio'].values
    mrr = best['validation_mrr'].values / baseline_mrr

    plt.plot(compression_ratio, mrr, label='Factorization')
    plt.legend(loc='lower right')
    plt.savefig('plot.svg')


if __name__ == '__main__':

    results = Results('results.txt')
    summarise_results(results)
