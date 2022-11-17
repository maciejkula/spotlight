### This script contains class and methods that processes.
## This script will download, merge and process data, and outputs ready-to-train movielens data in gzip format.
import requests, tempfile, os, zipfile, gzip, shutil
import pandas as pd


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
