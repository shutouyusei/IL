import pandas as pd
import os
from pathlib import Path


class ReadHDF:
    def __init__(self):
        self.KEY_NAME = 'sync_data'

    def __find_fdf(self,foldername):
        p = Path("datasets/" + foldername)
        hdf_files = list(p.glob("*.h5"))
        return hdf_files

    def read_hdf(self,foldername):
        hdf_files = self.__find_fdf(foldername)

        df_synced_list = []
        for file in hdf_files:
            try:
                sync_data = pd.read_hdf(file, key=self.KEY_NAME)
                sync_data['file_path'] = file
                df_synced_list.append(sync_data)
            except KeyError:
                print(f"ERROR:{file} does not have {self.KEY_NAME} key")
            except Exception as e:
                print("failed to read", file, e)
        return df_synced_list

if __name__ == '__main__':
    print("test")
    import argparse
    parser = argparse.ArgumentParser(description="read_hdf")
    parser.add_argument(
        '-f',
        '--foldername',
        required=True,
        dest='foldername',
        help="folder name"
    )
    args = parser.parse_args()
    read_hdf = ReadHDF()
    df_synced_list = read_hdf.read_hdf(args.foldername)
    print(df_synced_list[0].iloc[0])
