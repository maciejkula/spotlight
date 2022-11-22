### This script contains class and methods that processes.
## This script will download, merge and process data, and outputs ready-to-train movielens data in gzip format.
import requests, tempfile, os, zipfile, gzip, shutil
import pandas as pd
from config import DATA_PATH

class MovieLens:
    """
    This class is for Movielens data.
    The Movielens data is downloaded, processed, gzipped
    in this class.
    """
    URL_DATA = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    URL_DEMO = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    def __init__(
            self,
            use_demo_data:bool=False,
            use_big_data:bool=True,
            temp_folder=None,
            data_download_path = None,
            merged_data_path = None,
            output_path=None,
            compress=True
    ):
        self.use_demo_data = use_demo_data
        self.use_big_data = use_big_data
        if all([use_big_data, use_demo_data]): ## make sure not both are selected
            raise Exception("Cannot download both demo and big data at the same time")

        self.tempfolder = tempfile.mkdtemp(prefix="movielens") if temp_folder is None else temp_folder
        self.data_path = tempfile.mktemp(prefix="movielens_data", suffix=".zip", dir=self.tempfolder) if data_download_path is None else data_download_path
        self.data_merged = tempfile.mktemp(prefix="movielens_merged", suffix=".csv", dir=self.tempfolder) if merged_data_path is None else merged_data_path

        self.compress = compress
        self.output_path = output_path

        self.zipped_folder_name = None ##the files extracted to a folder ---> get from url, while downloading

    def process(self):
        self.download_data()
        self.unzip_data()
        self.merge_file()
        if self.compress:
            self.compress_data(self.data_merged, self.output_path)
        else:
            os.replace(self.data_merged, self.output_path)
        self.remove_temp(self.tempfolder)

    def download_data(self):
        ### downloads data from movielens link to a zip file
        url = self.URL_DATA if self.use_big_data and not self.use_demo_data else self.URL_DEMO
        self.zipped_folder_name = url.split("/")[-1].split(".")[0] ## To get the folder name from the url
        req = requests.get(url)
        with open(self.data_path, 'wb') as output_file:
            output_file.write(req.content)

    def unzip_data(self):
        ### unzips and extracts the files inside the zip
        with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
            zip_ref.extractall(self.tempfolder)

    def merge_file(self):
        ### merge the csv files
        files_to_merged = ["links.csv", "movies.csv", "ratings.csv", "tags.csv"]
        links_data = pd.read_csv(os.path.join(self.tempfolder, self.zipped_folder_name, "links.csv"))
        movies_data = pd.read_csv(os.path.join(self.tempfolder, self.zipped_folder_name, "movies.csv"))
        ratings_data = pd.read_csv(os.path.join(self.tempfolder, self.zipped_folder_name, "ratings.csv"))
        tags_data = pd.read_csv(os.path.join(self.tempfolder, self.zipped_folder_name, "tags.csv"))

        merged_data = pd.merge(
            pd.merge(
                pd.merge(movies_data, ratings_data, on="movieId", how="outer"), #1
                tags_data, on=["userId", "movieId", "timestamp"], how="outer"), #2
            links_data, on="movieId", how="outer" #3
        )

        merged_data.to_csv(self.data_merged, quoting=False, index=False)

    def process_data(self):
        pass

    def upload_data(self):
        pass

    def save_data(self):
        pass

    @staticmethod
    def compress_data(infile, outfile):
        import gzip
        with open(infile, 'rb') as f_in, gzip.open(f"{outfile}.gz", 'wb') as f_out:
            f_out.writelines(f_in)

    @staticmethod
    def remove_temp(path):
        ### removes a path and children underneath
        shutil.rmtree(path=path)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

class VisualizeData:
    """
    This class will visualize the characteristics of input data
    output is a pdf file with all the figures
    data_file: csv file
    out_pdf: output path for the pdf file. default data folder
    """

    data_folder = DATA_PATH

    def __init__(self, data_file:str, out_pdf:str=None):
        ###Files
        self.data_file = data_file
        self.out_pdf = out_pdf if out_pdf is not None else os.path.join(self.data_folder, "data_summary.pdf")

        ### declare pandas dataframe
        self.dataframe:pd.DataFrame = None

        ####Plotted variables, either matplot or seaborn plots
        self.plot_desc = None
        self.plot_nans = None
        self.plot_skew = None
        self.plot_kurt = None
        self.plot_corr = None

        ### Add to a dictionary that collects what to be plotted
        self.plot_variables = {}

    def process(self):
        self.read_data()
        self.plot_numeric_data(keys=["rating", "timestamp"])
        self.plot_category_counts()
        self.plot_data_distribution()
        self.plot_null()
        self.plot_corr_matrix()
        self.save_to_pdf()
    def read_data(self):
        """
        reads the data from 'data_file' and converts it into a pandas dataframe
        """
        if self.data_file.endswith(".gz") or self.data_file.endswith(".gzip"):
            self.dataframe = pd.read_csv(self.data_file, compression="gzip", sep=",")
        else:
            self.dataframe = pd.read_csv(self.data_file, sep=",")

    def plot_data_distribution(self):
        ### get data description
        #data_desc = self.dataframe.describe().reset_index()
        data_desc = self.dataframe.loc[:, ["rating", "timestamp", "userId", "movieId", "imdbId", 'tmdbId']]
        skew = {}
        kurt = {}
        for i in data_desc:  # to skip columns for plotting
            skew[i] = data_desc[i].skew()
            kurt[i] = data_desc[i].kurt()

        fig_skew = plt.figure()
        plt.plot(list(kurt.keys()), list(kurt.values()))
        #plt.xticks(rotation=45, horizontalalignment='right')
        plt.title("Skew Plot")
        self.plot_variables["skew_plot"] = fig_skew

        fig_kurt = plt.figure()
        plt.plot(list(skew.keys()), list(skew.values()))
        #plt.xticks(rotation=45, horizontalalignment='right')
        plt.title("Kurt Plot")
        self.plot_variables["kurt_plot"] = fig_kurt




    def plot_corr_matrix(self):
        corrmat = self.dataframe.corr()
        # %%
        fig = plt.figure()
        sns.heatmap(corrmat, vmax=1, annot=True, linewidths=.5)
        plt.xticks(rotation=30, horizontalalignment='right')
        plt.yticks(rotation=30, horizontalalignment='right')
        plt.title("Correlation Heatmap")

        ### Add variable dictionary
        self.plot_variables["correlation_matrix"] = fig

    def plot_null(self):
        ### Identify null_df
        ##TODO: @Thorstan can implement code here

        null_df = self.dataframe.apply(lambda x: sum(x.isnull())).to_frame(name="count")
        fig = plt.figure()
        plt.bar(null_df.index, null_df['count'])
        plt.title("NaN Values")
        plt.xticks(null_df.index, null_df.index, rotation=45, horizontalalignment='right')
        plt.xlabel("column names")
        plt.ylabel("count")
        plt.margins(0.1)
        ### Add to a dictionary that collects what to be plotted
        self.plot_variables["NaN"] = fig

    def plot_numeric_data(self, keys=None):
        for clm in keys:
            fig = plt.figure()
            self.dataframe.loc[:, [clm]].boxplot()
            plt.title(f"{clm} Values")
            self.plot_variables[clm] = fig

    def plot_category_counts(self): ##TODO: @Alessia can implement code here
        pass

    def save_to_pdf(self):
        # Create the PdfPages object to which we will save the pages:
        # Gets the figures from the "self.plot_variable" dictionary
        # Prints each figure to a pdf.
        with PdfPages(self.out_pdf) as pdf:
            if self.plot_variables is not None:
                for var_name, var_plot in self.plot_variables.items():
                    pdf.savefig(var_plot)

VisualizeData(data_file=os.path.join(DATA_PATH, "merged_ml_demo_data.csv.gz")).process()