import os

import requests


DATA_DIR = os.path.join(os.path.expanduser('~'),
                        'spotlight_data')


def create_data_dir(path):

    if not os.path.isdir(path):
        os.makedirs(path)


def download(url, dest_path, data_dir=DATA_DIR):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def get_data(url, dest_subdir, dest_filename, download_if_missing=True):

    data_dir = os.path.join(os.path.abspath(DATA_DIR), dest_subdir)

    create_data_dir(data_dir)

    dest_path = os.path.join(data_dir, dest_filename)

    if not os.path.isfile(dest_path):
        if download_if_missing:
            download(url, dest_path)
        else:
            raise IOError('Dataset missing.')

    return dest_path
