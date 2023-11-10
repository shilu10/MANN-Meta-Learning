from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse
import os 

def main(config):
    path = config.path 
    gdd.download_file_from_google_drive(
            file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
            dest_path="./omniglot_resized1.zip",
            unzip=True,
        )
    os.system('rm omniglot_resized1.zip')
    os.system('mv omniglot_resized1 data')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/')
    main(parser.parse_args())
