import argparse

import pandas as pd

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt

import seaborn as sns

from example import Results


def process_results(results):

    data = pd.DataFrame([x for x in results])

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

    return compression_ratio, mrr, elapsed


def plot_results(model, movielens, amazon):

    sns.set_style("darkgrid")

    for name, result in (('Movielens',
                          movielens), ('Amazon', amazon)):

        (compression_ratio,
         mrr,
         elapsed) = process_results(result)

        plt.plot(compression_ratio, mrr,
                 label=name)

    plt.ylabel("MRR ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs MRR ratio")

    plt.legend(loc='lower right')
    plt.savefig('{}_plot.png'.format(model))
    plt.close()

    for name, result in (('Movielens',
                          movielens), ('Amazon', amazon)):

        (compression_ratio,
         mrr,
         elapsed) = process_results(result)

        plt.plot(compression_ratio, elapsed,
                 label=name)

    plt.ylabel("Time ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs time ratio")

    plt.plot(compression_ratio, elapsed)
    plt.savefig('{}_time.png'.format(model))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)

    args = parser.parse_args()

    plot_results(args.model,
                 Results('movielens_{}_results.txt'.format(args.model)),
                 Results('amazon_{}_results.txt'.format(args.model)))
