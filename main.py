import argparse
import src.preprocess
import src.agent
import pandas as pd

def Arguments():
  parser = argparse.ArgumentParser(description="Backtest & Train AI Neural Network")
  parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
  parser.add_argument("-T", "--train", action="store_true", help="Enable Training Mode")
  parser.add_argument("-t", "--test", action="store_true", help="Enable Testing Mode")
  return parser.parse_args()

def Train(tData: pd.DataFrame, debug=False):
  agent = src.agent.Agent(tData, plot=True, debug=debug)
  agent.run(1000)

def Test():
  pass

def Main():
  args = Arguments()
  eurusd = pd.read_csv("./data/EURUSD.csv", index_col="Gmt time")
  gbpusd = pd.read_csv("./data/GBPUSD.csv", index_col="Gmt time")
  dataA, dataB = src.preprocess.Process_Test_Data(eurusd, gbpusd, debug=args.debug)

  if args.train:
    print("training mode")
    tData = src.preprocess.Process_Test_Data(dataA, dataB, args.debug)
    Train(tData, debug = args.debug)
  if args.test:
    print("testing mode")
    Test(debug = args.debug)

if __name__ == "__main__":
  try:
    Main()
  except Exception as e:
    print(f"Error running main function: {e}")
