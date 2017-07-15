import pandas as pd

from tabulate import tabulate


def _load_data(filename, columns=None):

    data = pd.read_json(filename, lines=True)
    data = data.sort_values('validation_mrr', ascending=False)

    mrr_cols = ['validation_mrr', 'test_mrr']

    if columns is None:
        columns = [x for x in data.columns if
                   (x not in mrr_cols and x != 'hash')]

    cols = data.columns
    cols = mrr_cols + columns

    return data[cols]


def _print_df(df):

    print(tabulate(df, headers=df.columns,
                   showindex=False,
                   tablefmt='pipe'))


def print_data():

    cnn_data = _load_data('results/cnn_results.txt',
                          ['residual',
                           'nonlinearity',
                           'loss',
                           'num_layers',
                           'kernel_width',
                           'dilation',
                           'embedding_dim'])
    _print_df(cnn_data[:5])

    lstm_data = _load_data('results/lstm_results.txt')

    _print_df(lstm_data[:5])

    pooling_data = _load_data('results/pooling_results.txt')

    _print_df(pooling_data[:5])


if __name__ == '__main__':
    print_data()
