import argparse
from pathlib import Path
from utils import prepare_amazon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prepare Amazon dataset")
    parser.add_argument("--input_dir",type=str, default="/home/anhvn/Master_Thesis_Resys/data/amazon")
    parser.add_argument("--input_fname",type=str, default="reviews_Movies_and_TV_5.json.gz")
    args = parser.parse_args()
    DATA_PATH = Path(args.input_dir)
    reviews = args.input_fname
    prepare_amazon(DATA_PATH, reviews)