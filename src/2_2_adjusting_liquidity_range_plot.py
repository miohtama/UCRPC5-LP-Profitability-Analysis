import pandas as pd
import matplotlib.pyplot as plt
import config  # Assuming the functions are available in the config module

# List of pools and respective winning addresses (multiple addresses per pool)
pools_and_winners = {
    'usdc-eth-005': ['0x58d797263f86ae8885c8757efe1a59bd179357c9'],
}

# Base path for the CSV files
csv_base_path = 'output/csv/2_2/merged_position_address_liquidity/'

# Function to plot converted prices for upper and lower ticks, volatility, and price for a specific pool and winner
def plot_prices_for_winner(pool, address):
    # Construct the path to the CSV file for the given pool
    csv_path = f'{csv_base_path}{pool}_merged.csv'
    sigma_csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/sigma.csv'
    
    # Read the CSV file for positions
    df = pd.read_csv(csv_path)

    # Read the CSV file for sigma and mean price
    sigma_df = pd.read_csv(sigma_csv_path)

    # Filter rows where the 'address' column matches the given address
    filtered_df = df[df['address'] == address]

    # Sort the DataFrame by the 'day' column
    filtered_df = filtered_df.sort_values(by='day')

    # Convert 'blk_time' to datetime format (if it's not already in the correct format)
    filtered_df['blk_time'] = pd.to_datetime(filtered_df['blk_time'], format='%Y-%m-%d %H:%M:%S')
    filtered_df['date'] = filtered_df['blk_time'].dt.date

    # Convert upper and lower ticks to prices using the config module
    filtered_df['price_upper'] = filtered_df['upper_tick'].map(config.fun)
    filtered_df['price_lower'] = filtered_df['lower_tick'].map(config.fun)

    # Ensure sigma_df also has 'blk_time' converted to datetime
    sigma_df['time'] = pd.to_datetime(sigma_df['time'], format='%Y-%m-%d')
    sigma_df['date'] = sigma_df['time'].dt.date

    # Create the figure and axis for the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot the mean_price on the same axis as price_upper and price_lower
    ax1.plot(sigma_df['date'], sigma_df['mean_price'], color='purple', alpha=0.7, label='Price (mean_price)', zorder=5)

    # Plot the vertical lines between 'price_lower' and 'price_upper' with reduced alpha
    for idx, row in filtered_df.iterrows():
        # Plot the vertical line
        ax1.plot([row['date'], row['date']], [row['price_lower'], row['price_upper']], color='blue', alpha=0.5, linestyle='--')  # Decreased alpha to 0.5

        # Plot the circles at both ends
        ax1.scatter(row['date'], row['price_upper'], color='blue', s=10)  # Small circle for price_upper
        ax1.scatter(row['date'], row['price_lower'], color='blue', s=10)  # Small circle for price_lower

    ax1.plot([row['date'], row['date']], [row['price_lower'], row['price_upper']], color='blue', alpha=0.5, linestyle='--', label='Price/Tick Ranges')

    # Set labels and title for the first y-axis (prices)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price Value')
    ax1.set_title(f'Price Range, Volatility, and Price Over Time - Top Performer (16% net return) of Pool {pool}')
    ax1.tick_params(axis='y')

    # Create the second y-axis for volatility (sigma_ln_r) with dashed line and 0.8 alpha
    ax2 = ax1.twinx()
    ax2.plot(sigma_df['date'], sigma_df['sigma_ln_r'], color='orange', linestyle='--', alpha=0.8, label='Volatility (sigma_ln_r)')  # Dashed line and alpha=0.8
    ax2.set_ylabel('Volatility (sigma_ln_r)')
    ax2.tick_params(axis='y')

    # Add legends for all axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot to a file
    output_path = f'output/img/2_2/LP_range_pattern/{pool}_{address}.png'
    plt.tight_layout()
    plt.savefig(output_path)

    # Print confirmation message
    print(f"Scatter plot for {pool} (address: {address}) saved to {output_path}")
    plt.close()

# Iterate over each pool and corresponding winner addresses
for pool, addresses in pools_and_winners.items():
    for address in addresses:  # Loop through all addresses for the pool
        plot_prices_for_winner(pool, address)
