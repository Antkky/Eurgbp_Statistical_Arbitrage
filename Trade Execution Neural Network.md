
## Processed Dataframe

- EURUSD Price
- GBPUSD Price
- Ratio Price
- Hedge Ratio
- Rolling Spread-ZScore
- Rolling Correlation
- Rolling Cointegration Score
- EURUSD Relative Strength Index
- GBPUSD Relative Strength Index
- Ratio Relative Strength Index
- EURUSD Moving Average Convergence Divergence
- GBPUSD Moving Average Convergence Divergence
- Ratio Moving Average Convergence Divergence

make sure to normalize the data using the min-max scaler

# Inputs

- Rolling Spread-ZScore
- Rolling Correlation
- Rolling Cointegration Score
- Relative Strength Index
- Moving Average Convergence Divergence
- Current Position Size
- Current PnL
# Outputs

- Hold
- Buy
- Sell
- Flatten


### Environment

The environment is the place where the AI will train as it will simulate a real market, it is able to place trades, manage trades and realize trades

i want the environment to output every single training epoch and episode into a CSV file so i can later on analyze the post training data