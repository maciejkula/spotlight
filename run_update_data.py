#### This script contains the main run function ---> CLI

def execute(arg):
    from process_data.process_movie_data import MovieLens
    from config import DATA_PATH
    import os
    outpath = arg.outpath if arg.outpath is not None else os.path.join(DATA_PATH, "ml-data.csv")
    use_demo = arg.demo_file
    MovieLens( ## initialize the class
        use_demo_data = use_demo,
        use_big_data = not use_demo,
        data_download_path = None,
        merged_data_path = None,
        output_path = outpath,
        compress = arg.compress
    ).process() ## process the algorithm



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    ###Input files
    parser.add_argument('--demo-file', dest="demo_file", help="downloads and uses small demo dataset", required=False,
                        action="store_true")
    parser.add_argument('--outfile', dest='outpath', help='path to outputfile',
                        required=False, default=None)
    parser.add_argument("--compress", dest="compress", help="compress output file to gzip format", required=False,
                        action="store_true")

    args = parser.parse_args()
    execute(args)