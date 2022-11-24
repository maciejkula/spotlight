
def execute(args):
    from process_data.process_movie_data import VisualizeData
    from config import DATA_PATH
    import os

    VisualizeData(
        data_file=os.path.join(DATA_PATH, "ml-data.csv.gz") if args.infile is None else args.infile,
        out_pdf=args.outpath if args.outpath is not None else os.path.join(DATA_PATH, "data_summary.pdf"),
        numeric_data_keys=args.num_col if args.num_col is not None else ["rating", "timestamp"]
    ).process()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    ###Input files
    parser.add_argument('--data-file', dest="infile", help="input dataset", required=False,
                        default=None)
    parser.add_argument('--outfile', dest='outpath', help='path to outputfile',
                        required=False, default=None)
    parser.add_argument('--numerical-columns', dest="num_col", help="data column names that contain numeric data that you want to plot",
                        required=False, default=None, nargs="+")

    args = parser.parse_args()
    execute(args)