import pandas as pd

LOG_FILE = "logs/signal_log.csv"

def summarize_signal_log(log_path=LOG_FILE):
    df = pd.read_csv(log_path)
    df = df.dropna(subset=["profit"])
    df["profit"] = df["profit"].astype(float)

    win_df = df[df["profit"] > 0]
    loss_df = df[df["profit"] < 0]

    print("📊 Summary of Spread Reversion Trades:")
    print("──────────────────────────────────────────────")
    print(f"Total Trades:         {len(df)}")
    print(f"Profitable Trades:    {len(win_df)}")
    print(f"Loss Trades:          {len(loss_df)}")
    print(f"Win Rate:             {len(win_df) / len(df):.2%}")
    print(f"Average Profit:       {df['profit'].mean():.4f}")
    print(f"Max Profit:           {df['profit'].max():.4f}")
    print(f"Max Loss:             {df['profit'].min():.4f}")
    print("──────────────────────────────────────────────")

if __name__ == "__main__":
    summarize_signal_log()
