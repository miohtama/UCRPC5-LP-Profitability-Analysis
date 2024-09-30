import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np
from datetime import timedelta
import config
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt


def makedirs_clean (folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)


def address_list_iter(pool):
    #return a list
    lp_folder = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'
    return os.listdir(lp_folder)



# function to get capm



def get_return_rate(pool, address):
    df = pd.read_csv(f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result/{address}.csv', index_col=0)
    #index as datetime
    df.index = pd.to_datetime(df.index)
    return df["return_rate"]

def get_eth_price():
    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df_price = pd.read_csv(csv_path, low_memory=False)
    df_price['block_timestamp'] = pd.to_datetime(df_price['block_timestamp'])
    #resample to hourly
    df_price = df_price.set_index('block_timestamp').resample('H').last()
    df_price = df_price.fillna(method='ffill')
    return df_price["price"]



def get_capm(pool,eth_price_series,address):
    lp_return_rate = get_return_rate(pool, address)
    lp_return_rate=lp_return_rate[:-1]
    lp_return_rate = lp_return_rate.resample('H').last()
    lp_return_rate = lp_return_rate.fillna(method='ffill') 
    # return rate -1
    lp_return_rate = lp_return_rate - 1
    #Capital Asset Pricing Model, eth_price is the market return rate
    #risk free rate is 0.00
    #market return rate is eth_price
    #return rate is cumulate_return_rate
    # drop eth by return rate index
    eth_price_joined = eth_price_series[lp_return_rate.index]
    eth_return = eth_price_joined.pct_change()
    eth_return = eth_return.fillna(0)
    #cov(eth_price, cumulate_return_rate)
    beta = np.cov(eth_return, lp_return_rate)[0][1] / np.var(eth_return)
    alpha = np.mean(lp_return_rate) - beta * np.mean(eth_return)
    yearly_alpha = alpha * 365 * 24
    return yearly_alpha, beta


def calculate_capm(pool, address_list):
    eth_price = get_eth_price()
    capm_result=[]
    for address in tqdm(address_list):
        try:
            alpha, beta = get_capm(pool,eth_price, address)
            #print(f'address: {address}, alpha: {alpha}, beta: {beta}')
            capm_result.append([address, alpha, beta])
        #print exception
        except Exception as e:
            print(e)
            continue
    capm_result = pd.DataFrame(capm_result, columns=['address', 'alpha', 'beta'])
    capm_result.to_csv(f"capm_{pool}_hou.csv", index=False)

def capm_demo():
    pool_list = ["usdc-eth-005"]
    pool = pool_list[0]
    address_list = address_list_iter(pool)
    address_list = [ add.split(".")[0] for add in address_list]
    calculate_capm(pool, address_list)


def plot_capm_and_save(pool,address_list):
    capm = pd.read_csv(f"capm_{pool}_hou.csv")
    # 
    capmx = capm[capm["address"].isin(address_list)]
    capmx = capmx[capmx["beta"]>0]
    capmx = capmx[capmx["beta"]<3]
    #capmx = capmx.sort_values(by='alpha', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(capmx['beta'], capmx['alpha'], s=1)
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.title(f'CAPM for {pool}')
    plt.savefig(f'capm_{pool}_hou.png')

def print_top_alpha(pool,address_list):
    capm = pd.read_csv(f"capm_{pool}_hou.csv")
    capmx = capm[capm["address"].isin(address_list)]
    capmx = capmx[capmx["beta"]>0]
    capmx = capmx.sort_values(by='alpha', ascending=False)
    print("pool: ",pool)
    print(capmx.head(10))


#use filter data from pervious result
if __name__ == '__main__':
    # filter_address = list(pd.read_csv("./output/csv/1_4/usdc-eth-005.csv")["Name"])   
    # plot_capm_and_save("usdc-eth-005",filter_address)

    pool_list = ["usdc-eth-001","usdc-eth-005","usdc-eth-03"]
    pool = pool_list[0]
    address_list = address_list_iter(pool)
    address_list = [ add.split(".")[0] for add in address_list]
    #calculate_capm(pool, address_list)
    filter_address = list(pd.read_csv(f"./output/csv/1_4/{pool}.csv")["Name"])
    print_top_alpha(pool,filter_address)