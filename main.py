import argparse
import pandas as pd
import sys
from src.preprocess import process_test_data, process_train_data
from src.agent import TradingAgent

def get_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Backtest & Train AI Neural Network")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-T", "--train", action="store_true", help="Enable Training Mode")
    parser.add_argument("-t", "--test", action="store_true", help="Enable Testing Mode")
    return parser.parse_args()

def train(tData: pd.DataFrame, debug=False):
    """Trains the AI model."""
    agent = TradingAgent(tData, plot=True, debug=debug)
    agent.run(1000)

def test():
    """Placeholder for testing mode logic."""
    print("Testing mode not implemented yet.")

def main():
    """Main function to handle training and testing."""
    args = get_arguments()
    #dataA, dataB = process_test_data("./data/EURUSD.csv", "./data/GBPUSD.csv", debug=args.debug)
    #dataT = process_train_data(dataA, dataB)

    if args.train:
        print("Training mode")
        dataT = pd.read_csv("./data/processed/train_data.csv", index_col="Gmt time")
        train(dataT, debug=args.debug)  # Processed dataA is directly used

    if args.test:
        print("Testing mode")
        test()

if __name__ == "__main__":
    main()
