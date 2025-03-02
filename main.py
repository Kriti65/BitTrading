import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

  # Import your existing functions

# Function to upload & map columns
def upload_and_map_csv():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Show preview of dataset
        st.write("Preview of Uploaded Data", df.head())

        # Expected columns for price and volume analysis
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]

        # User-defined column mapping
        user_mappings = {}
        st.subheader("Map Your Columns to Expected Names")
        for col in expected_columns:
            user_mappings[col] = st.selectbox(f"Select column for {col}", df.columns, index=0)

        # Rename columns based on user selection
        df = df.rename(columns=user_mappings)
        selected_columns = list(user_mappings.values())  # Get mapped column names
        df = df[selected_columns]

        # Keep only relevant columns
        #df = df[expected_columns]

        return df

    return None
def calcEMAs(df):
    
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['MACD_Line'] = df['EMA_20'] - df['EMA_50']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=18, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

    return df


def generateSignals(dataframe):
    signal = []
    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        percentage = 100 * (row['EMA_20'] - row['EMA_50']) / row['EMA_50']
        
        if percentage < 1.0 or abs(row['MACD_Histogram']) <= 15:
            signal.append("Hold")
        elif row['EMA_20'] > row['EMA_50'] and row['MACD_Line'] > row['Signal_Line']:
            signal.append("Buy")
        elif row['EMA_20'] < row['EMA_50'] and row['MACD_Line'] < row['Signal_Line']:
            signal.append("Sell")
        else:
            signal.append("Hold")
     
    return signal
def backtest(df, initial_amount, stop_loss_percentage, buy_fraction):
    amount = initial_amount
    units_held = 0
    stop_loss_threshold = amount * (1 - (stop_loss_percentage / 100))

    for i in range(len(df)):
        row = df.iloc[i]

        # Stop loss check
        if amount + units_held * row['Close'] <= stop_loss_threshold:
            st.warning(f"❌ Stop loss triggered at {row['Close']}")
            break

        # Buy logic
        if row['Signals'] == "Buy" and amount > 0:
            units_to_buy = (amount * buy_fraction) / row['Close']
            units_held += units_to_buy
            amount -= units_to_buy * row['Close']
            st.write(f"✅ Bought {units_to_buy:.4f} BTC at {row['Close']}, remaining cash: {amount:.2f}")

        # Sell logic
        elif row['Signals'] == "Sell" and units_held > 0:
            amount += units_held * row['Close']
            units_held = 0
            st.write(f"💰 Sold all BTC at {row['Close']}, remaining cash: {amount:.2f}")

    # Final portfolio value
    final_value = amount + (units_held * df.iloc[-1]['Close'])
    percentage_profit = 100 * (final_value - initial_amount) / initial_amount

    st.success(f"💰 **Final Amount:** ${final_value:.2f}")
    st.success(f"📈 **Total Profit:** {percentage_profit:.2f}%")

    return percentage_profit
def plot_performance(df):
    st.subheader("Portfolio Value Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label="Price")
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
def main():
    st.title("Trading Strategy Backtest with Custom CSV Data")

    # Sidebar parameters
    st.sidebar.header("Strategy Parameters")
    initial_amount = st.number_input("💰 Enter Initial Capital ($)", min_value=1000, value=25000, step=500)
    stop_loss_percentage = st.number_input("Stop Loss Percentage")
    buy_fraction = st.sidebar.slider("Buy Fraction", 0.05, 1.0, 0.25)

    # Upload CSV and map columns
    df = upload_and_map_csv()

    if df is not None:
        # Generate Buy/Sell signals
        df = calcEMAs(df)
        df['Signals'] = generateSignals(df)

        # Show Dataframe with Mapped Columns
        st.write("Processed Data", df.head())

        # Run Backtest
        if st.button("Run Backtest"):
            profit = backtest(df,initial_amount,stop_loss_percentage, buy_fraction)
            st.write(f"Total Profit: {profit}%")
            plot_performance(df)

# Plotting function


if __name__ == "__main__":
    main()