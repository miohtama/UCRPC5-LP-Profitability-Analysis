import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np
from datetime import timedelta
import config
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt

# pool_list = ["usdc-eth-03"]
pool_list = ["usdc-eth-001","usdc-eth-03","usdc-eth-005","wbtc-eth-03"]
# "usdc-eth-001","usdc-eth-03","usdc-eth-005",

def makedirs_clean (folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for root, dirs, files in os.walk(folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)

def tick_addr_pos(pool):
    lp_folder = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'
    pos_addr_file = f'../../../../data/research/task2403-uni-profitability/{pool}/3_position_address.csv'
    pos_liq_file = f'../../../../data/research/task2403-uni-profitability/{pool}/2_position_liquidity.csv'
    save_folder = f'output/csv/1_4/lp_addr_tick/{pool}'
    makedirs_clean (save_folder+"/init")
    makedirs_clean (save_folder+"/price_result")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    df_pos_addr = pd.read_csv(pos_addr_file,low_memory=False) 
    # position,address,day,tx
    df_pos_liq = pd.read_csv(pos_liq_file,low_memory=False) 
    #id,lower_tick,upper_tick,tx_type,block_number,tx_hash,log_index,blk_time,liquidity,final_amount0,final_amount1

    address_list = []
    for filename in os.listdir(lp_folder):
        if filename.endswith('.csv'):
            file_name = os.path.splitext(filename)[0]
            address_list.append(file_name)

    for name in tqdm(address_list, desc='Processing Files1'):

        path_result = os.path.join(save_folder+"/init", name + '.csv') 
        position_list = df_pos_addr[df_pos_addr['address'] == name]['position']
        df_addr_liq = df_pos_liq[df_pos_liq['id'].isin(position_list)]
        df_addr_liq.to_csv(path_result, index=False)

    for name in tqdm(address_list, desc='Processing Files2'):
        lp_addr_file = os.path.join(save_folder+"/init", name + '.csv')
        lp_addr_price_file = os.path.join(save_folder+"/price_result", name + '.csv')

        action = pd.read_csv(lp_addr_file, sep=',', dtype=object, header=0)
        action.loc[:, "blk_time"] = action.loc[:, "blk_time"].apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        action['tick_id'] = action['lower_tick']+ '-' + action['upper_tick']
        action[['final_amount0', 'final_amount1',  "lower_tick", "upper_tick",'liquidity']] = action[
            ['final_amount0', 'final_amount1',  "lower_tick", "upper_tick",'liquidity']].astype(float)
        action[["lower_tick", "upper_tick"]] = action[["lower_tick", "upper_tick"]].astype('Int64')
        idx = action[(action.tx_type == 'BURN')&(action.liquidity == 0)].index.values
        action = action.drop(idx)
        action = action.sort_values(['blk_time','tx_type'], ascending=[True,False])
        action = action.reset_index()
        action_id = action.groupby('tick_id')
        idx = np.array([])
        for id, df in action_id:
            df['cum_liq'] = df['liquidity'].cumsum()
            mint = np.array(df[df.tx_type == 'MINT'].index.values)
            if len(mint) == 0:
                continue
            burn0 = np.array(df[(df.tx_type == 'BURN') & (df.cum_liq <= 200)].index.values)
            burn0 = burn0[burn0>mint[0]]
            idx = np.append(idx, burn0)
            if len(burn0) == 0: # 0 burn
                idx = np.append(idx, mint[0])
            else:
                count = 0
                for i in burn0:
                    try:
                        idx = np.append(idx, mint[0])
                        action.loc[[mint[0], i], 'tick_id'] = action.loc[[mint[0], i], 'tick_id'] +\
                                                                    '-' + str(count)
                        mint = mint[mint>i]
                        count += 1
                    except IndexError:
                        print("IndexError occurred. Skipping this iteration.")
                        if name =="0xa173340f1e942c2845bcbce8ebd411022e18eb13":
                            print(len(action))
                        continue

                if mint.size > 0:
                    idx = np.append(idx, mint[0])
                    action.loc[mint[0], 'tick_id'] = action.loc[mint[0], 'tick_id'] + \
                                                                '-' + str(count)
        action = action.loc[idx,:]
        action.loc[:, 'price_upper'] = action.loc[:, 'lower_tick'].map(config.fun)
        action.loc[:, 'price_lower'] = action.loc[:, 'upper_tick'].map(config.fun)
        action.to_csv(lp_addr_price_file)
# for pool in pool_list:
#     print(f"================== Init addr_pos_tick: {pool} ==================")
#     tick_addr_pos(pool)

def cal_lp_info(pool):
    lp_folder = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'
    lp_addr_price_folder = f'output/csv/1_4/lp_addr_tick/{pool}/price_result'

    save_folder = f'output/csv/1_4/lp_info/{pool}'
    makedirs_clean (save_folder)
    data_result = pd.DataFrame(columns=['Name', 'Holding Time Per', 'Max Net Value','beta','alpha'])

    pos_paths = [os.path.join(lp_addr_price_folder, file) for file in os.listdir(lp_folder) if file.endswith('.csv')]
    address_list = []
    for file_path in tqdm(pos_paths, desc='read address list'):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        address_list.append(file_name)
    
    outRange = 0
    outRange2  = 0

    for address_name in tqdm(address_list, desc='Processing Files'):
        data_path = os.path.join(lp_folder, address_name+'.csv')
        # ,net_value,fee0,fee1,amount0,amount1,lp_value_with_prev_liq,positions,return_rate,cumulate_return_rate
        liq_path = os.path.join(lp_addr_price_folder, address_name+'.csv')
        # ,index,id,lower_tick,upper_tick,tx_type,block_number,tx_hash,log_index,blk_time,liquidity,final_amount0,final_amount1,tick_id,price_upper,price_lower
        try:
            df1 = pd.read_csv(data_path, index_col=["Unnamed: 0"])
        except EmptyDataError:
            print("empty")
            outRange2 +=1
            continue
        df2 = pd.read_csv(liq_path,dtype=object, header=0) 
        df1.index = pd.to_datetime(df1.index)
        df1 = df1[(df1.index >=  datetime.strptime('2024-03-15','%Y-%m-%d'))& (df1.index <= datetime.strptime('2024-09-15','%Y-%m-%d'))]
        df1 = df1[df1['net_value'] >= 1]
        if df1.empty:
            outRange2 +=1
            continue
        max_net_value = df1['net_value'].max()

        df2 = df2[df2.tx_type != 'COLLECT']
        # 修改数据格式
        df2.loc[:, "blk_time"] = df2.loc[:, "blk_time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        # print(type(action.loc[:,'block_timestamp'][0]), action.loc[:,'block_timestamp'][0])
        df2 = df2[df2.blk_time >= datetime.strptime('2024-03-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
        df2 = df2[df2.blk_time <= datetime.strptime('2024-09-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
        if df2.empty:
            outRange+=1
            continue

        df2.loc[:, "blk_time"] = df2.loc[:, "blk_time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))

        df2 = df2.sort_values(['blk_time', 'tx_type'], ascending=[True, False])
        df2.index = df2.loc[:, "blk_time"]
        df2.loc[:,'tick_id'] = df2.loc[:,'lower_tick']+df2.loc[:,'upper_tick']

        try:
            df2[['price_lower', 'price_upper','liquidity']] = df2[['price_lower', 'price_upper','liquidity']].astype(float)
        except:
            df2[['lower_tick', 'upper_tick','liquidity']] = df2[['lower_tick', 'upper_tick','liquidity']].astype(float)
            df2[['lower_tick', 'upper_tick']] = df2[['lower_tick', 'upper_tick']].astype('Int64')

        # 删掉burn liquidity=0的数据
        idx = df2[(df2.tx_type == 'BURN')&(df2.liquidity == 0)].index.values
        df2 = df2.drop(idx)

        df2_id = df2.groupby('tick_id')
        
        # 'holding_time','pos_inter_time','pos_inter_price','pos_num'
        pos_data = pd.DataFrame(columns=['id','holding_time','pos_inter_time','pos_inter_price','pos_num'])

        pos_num = 0
        holding_time = timedelta(hours=0)
        # 收集所有的start和end时间对
        formatted_periods = []
        for id, df in df2_id:
            a = 0
            try:
                df_mint = df[df.tx_type == 'MINT']
                start_hour = df_mint['blk_time'][0]
                a = 1
                price_up = df_mint['price_upper'][0]
                price_low = df_mint['price_lower'][0]
            except:
                start_hour = df2.loc[:, 'blk_time'].min() # 被时间截取错过的pos
                
            try:
                df_burn = df[df.tx_type == 'BURN']
                end_hour = df_burn['blk_time'][-1]
                if a == 0:
                    price_up = df_burn['price_upper'][-1]
                    price_low = df_burn['price_lower'][-1]
            except:
                end_hour = df2.loc[:, 'blk_time'].max()

            # 'holding_time','pos_inter_time','pos_inter_price','pos_num'
            start_hour = datetime.strptime(start_hour, "%Y-%m-%d %H:%M")
            end_hour = datetime.strptime(end_hour, "%Y-%m-%d %H:%M")
            pos_inter_time = end_hour-start_hour

            formatted_periods.append({"start": start_hour, "end": end_hour})

            pos_inter_price = price_up-price_low
            pos_num += 1
            holding_time += pos_inter_time
            new_row = {'id': id, 'holding_time':holding_time, 'pos_inter_time': pos_inter_time, 'pos_inter_price': pos_inter_price, 'pos_num': pos_num}
            pos_data = pos_data._append(new_row, ignore_index=True)

        pos_cal_file = os.path.join(save_folder, address_name + '.csv')
        pos_data.to_csv(pos_cal_file, index=False)
        formatted_periods.sort(key=lambda x: x["start"])
        total_duration = timedelta(0)
        previous_end = formatted_periods[0]["start"]
        for period in formatted_periods:
            if period["start"] > previous_end:
                total_duration += period["end"] - period["start"]
            elif period["end"] > previous_end:
                total_duration += period["end"] - previous_end
            previous_end = max(previous_end, period["end"])

        s_time = datetime.strptime('2024-03-15 00:00:00', '%Y-%m-%d %H:%M:%S')
        e_time = datetime.strptime('2024-09-15 00:00:00', '%Y-%m-%d %H:%M:%S')
        total_duration_per = (total_duration/(e_time-s_time))
        holding_time = pos_data.iloc[-1]['holding_time']
        pos_num = pos_data.iloc[-1]['pos_num']

        # 'Name', 'Holding Time Per', 'Max Net Value','Annual Return','Excess Rerturn','beta','alpha'
        data_result = data_result._append({'Name': address_name,'Holding Time Per': total_duration_per,'Max Net Value': max_net_value}, ignore_index=True)
    print(outRange2)
    print(outRange)
    new_csv_path = f'output/csv/1_4/{pool}.csv'
    data_result.to_csv(new_csv_path, index=False)  
# for pool in pool_list:
#     print(f"================== Get Info of LP: {pool} ==================")
#     cal_lp_info(pool)

def filter_address(pool):
    lp_file = f'output/csv/1_4/{pool}.csv'
    filter_file = f'output/csv/1_4/filtered/{pool}.csv'
    df = pd.read_csv(lp_file)
    
    df['Holding Time Per'] = df['Holding Time Per'].astype(float)
    df['Max Net Value'] = df['Max Net Value'].astype(float)

    original_count = len(df)
    print(f"Total addresses before filtering: {len(df)}")
    df = df[df['Holding Time Per'] >= 0.5]
    holding_time_filtered_count = original_count - len(df)
    print(f"Filtered by Holding Time Per >= 0.5: {holding_time_filtered_count} addresses excluded")
    
    df = df[df['Max Net Value'] >= 1000]
    max_net_value_filtered_count = original_count - holding_time_filtered_count - len(df)
    print(f"Filtered by Max Net Value >= 1000: {max_net_value_filtered_count} addresses excluded")

    df.to_csv(filter_file, index=False)
    print(f"Total addresses after filtering: {len(df)}")
# for pool in pool_list:
#     print(f"================== Filtering LP: {pool} ==================")
#     filter_address(pool)

pool_list_10_lp = ["usdc-eth-001","usdc-eth-03","usdc-eth-005"]
def capm_market(pool):
    makedirs_clean(f'output/csv/1_4/capm_process_merge/{pool}')
    lp_folder = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'
    filter_file = f'output/csv/1_4/filtered/{pool}.csv'
    info_df = pd.read_csv(filter_file,index_col=0)

    # market
    df_mk = pd.read_csv(f'output/csv/1_2/lp_profit/{pool}_lp_profit.csv', sep=',', index_col=[],  header=0)
    # date,weighted_mean_return
    df_mk['date'] = pd.to_datetime(df_mk['date'])

    # eth
    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df_price = pd.read_csv(csv_path, low_memory=False)
    df_price['block_timestamp'] = pd.to_datetime(df_price['block_timestamp'])
    df_price['date'] = df_price['block_timestamp'].dt.date
    daily_prices = df_price.groupby('date')['price'].last().reset_index()
    daily_prices = daily_prices.reset_index(drop=True)
    daily_prices['price_pct'] = daily_prices['price'].pct_change()+1
    daily_prices['cumulative_price_pct'] = (daily_prices['price_pct']).cumprod()
    daily_prices['date'] = pd.to_datetime(daily_prices['date'])

    result_df = pd.DataFrame(columns=['address', 'alpha_market', 'beta_market', 'alpha_eth', 'beta_eth'])
    no_alpha = 0
    
    for address_name in tqdm(info_df.index, desc='Processing Files'):
        df = pd.read_csv(lp_folder+"/"+address_name+".csv")
        df['hour'] = pd.to_datetime(df.iloc[:, 0])
        df['date'] = df['hour'].dt.date
        daily_data = df.groupby('date').apply(lambda x: x.iloc[-1])[['date', 'cumulate_return_rate', 'net_value']]
        daily_data['address'] = address_name
        
        start_date = datetime.strptime('2024-03-15', '%Y-%m-%d').date()
        end_date = datetime.strptime('2024-09-15', '%Y-%m-%d').date()
        daily_data = daily_data[(daily_data['date'] >= start_date) & (daily_data['date'] <= end_date)]
        daily_data = daily_data[daily_data['net_value'] >= 1]
        daily_data = daily_data.reset_index(drop=True)
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        merged_df = pd.merge(daily_data, df_mk, on='date', how='inner')

        merged_df2 = pd.merge(merged_df, daily_prices, on='date', how='inner')

        merged_df2.to_csv(f'output/csv/1_4/capm_process_merge/{pool}/{address_name}.csv', index=False)

        ri = merged_df2['cumulate_return_rate'].values  # 资产收益率
        rm_m = merged_df2['weighted_mean_return'].values  # 市场收益率 market
        rm_e = merged_df2['price'].values  # 市场收益率 eth
        ri = np.nan_to_num(ri, nan=1.0, posinf=1.0, neginf=1.0)
        rm_m = np.nan_to_num(rm_m, nan=1.0, posinf=1.0, neginf=1.0)
        rm_e = np.nan_to_num(rm_e, nan=1.0, posinf=1.0, neginf=1.0)
        ri = ri.astype('float64')
        rm_m = rm_m.astype('float64')
        rm_e = rm_e.astype('float64')

        Xm = np.vstack([rm_m, np.ones(len(rm_m))]).T
        Xm = Xm.astype('float64')
        Xe = np.vstack([rm_e, np.ones(len(rm_e))]).T
        Xe = Xe.astype('float64')
        try:
            betam,alpham= np.linalg.lstsq(Xm, ri, rcond=None)[0]
            betae,alphae= np.linalg.lstsq(Xe, ri, rcond=None)[0]
            result_df = result_df._append({'address': address_name,  'alpha_market': alpham, 'beta_market': betam, 'alpha_eth': alphae, 'beta_eth': betae}, ignore_index=True)
        except:
            no_alpha += 1
            continue
    # ri_e = merged_df2['cumulative_price_pct'].values  # eth
    # rm_m = merged_df2['weighted_mean_return'].values  # 市场收益率
    # ri_e = np.nan_to_num(ri_e, nan=1.0, posinf=1.0, neginf=1.0)
    # rm_m = np.nan_to_num(rm_m, nan=1.0, posinf=1.0, neginf=1.0)
    # Xie = np.vstack([rm_m, np.ones(len(rm_m))]).T
    # Xie = Xie.astype('float64')
    # beta,alpha= np.linalg.lstsq(Xie, ri_e, rcond=None)[0]
    # print(f"ETH Beta related to Market: {beta}")
    # print(f"ETH Alpha related to Market: {alpha}")
    # result_df = result_df._append({'address': 'eth', 'alpha_market': alpha, 'beta_market': beta,'alpha_eth':"self", 'beta_eth': "self"}, ignore_index=True)

    # ri_m = merged_df2['weighted_mean_return'].values  # market
    # rm_e = merged_df2['cumulative_price_pct'].values  # 市场收益率
    # ri_m = np.nan_to_num(ri_m, nan=1.0, posinf=1.0, neginf=1.0)
    # rm_e = np.nan_to_num(rm_e, nan=1.0, posinf=1.0, neginf=1.0)
    # Xim = np.vstack([rm_e, np.ones(len(rm_e))]).T
    # Xim = Xim.astype('float64')
    # beta,alpha= np.linalg.lstsq(Xim, ri_m, rcond=None)[0]
    # print(f"Market Beta related to ETH: {beta}")
    # print(f"Market Alpha related to ETH: {alpha}")
    # result_df = result_df._append({'address': 'market', 'alpha_market': "self", 'beta_market': "self",'alpha_eth':alpha, 'beta_eth': beta}, ignore_index=True)

    new_file_name = f'output/csv/1_4/capm_market/{pool}.csv'
    result_df.to_csv(new_file_name, index=False)
    print(no_alpha)
# for pool in pool_list_10_lp:
#     print(f"================== CAPM: {pool} ==================")
#     capm_market(pool)

def capm_plot(pool):
    df_capm = pd.read_csv(f'output/csv/1_4/capm_market/{pool}.csv', sep=',', index_col=[],  header=0)

    plt.figure(figsize=(10, 6))
    plt.scatter(df_capm['beta_eth'], df_capm['alpha_eth'], alpha=0.5, label='ETH as market')
    # plt.scatter(df_capm['beta_market'], df_capm['alpha_market'], alpha=0.5, color='red', label='other LPs as market')
    
    # market_row = df_capm[df_capm['address'] == 'market']
    # eth_row = df_capm[df_capm['address'] == 'eth']

    # # 特殊化处理 market 和 eth 的点
    # plt.scatter(market_row['beta_eth'], market_row['alpha_eth'], marker='*', s=200, color='green', label='market alpha relate to eth')
    # plt.scatter(eth_row['beta_market'], eth_row['alpha_market'], marker='*', s=200, color='purple', label='eth alpha relate to market')

    # plt.annotate('market alpha relate to eth', (market_row['beta_eth'], market_row['alpha_eth']), textcoords="offset points", xytext=(10,10), ha='center')
    # plt.annotate('eth alpha relate to market', (eth_row['beta_market'], eth_row['alpha_market']), textcoords="offset points", xytext=(10,10), ha='center')


    plt.legend()
    plt.title(f'CAPM Analysis for {pool}')
    plt.xlabel('Beta (ETH)')
    plt.ylabel('Alpha (ETH)')

    plot_filename = f'output/img/1_4/{pool}_capm_plot.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
# for pool in pool_list_10_lp:
#     print(f"================== CAPM Plot: {pool} ==================")
#     capm_plot(pool)


def filter_10_alpha(pool):
    read_path = f'output/csv/1_4/capm_market/{pool}.csv'
    save_path = f'output/csv/1_4/capm_max_10/{pool}.csv'
    df = pd.read_csv(read_path)
    sorted_data = df.sort_values(by="alpha_eth", ascending=False)
    top_addresses = pd.DataFrame(columns=['address','alpha','beta'])
    top_addresses['address'] = sorted_data["address"].head(10)
    top_addresses['alpha'] = sorted_data["alpha_eth"].head(10)
    top_addresses['beta'] = sorted_data["beta_eth"].head(10)
    top_addresses.to_csv(save_path, index=False)
# for pool in pool_list_10_lp:
#     print(f"================== CAPM Find Max Alpha LPs: {pool} ==================")
#     filter_10_alpha(pool)

def visuallize_action_plot(pool):
    save_path = f'output/csv/1_4/lp_addr_tick/{pool}'
    save_path1 = f'output/img/1_4/visuallize_action/{pool}'
    makedirs_clean(save_path1)

    data_folder_path = f'../../../../data/research/task2403-uni-profitability/{pool}/6_address_result'

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df_info = pd.read_csv(f'output/csv/1_4/capm_max_10/{pool}.csv', sep=',', index_col=[], dtype=object, header=0)
    df_info.index = df_info['address']
    if 'name' not in df_info.columns:
        df_info['name'] = range(len(df_info))
    csv_path = f'../../../../data/research/task2403-uni-profitability/{pool}/4_price.csv'
    df_price = pd.read_csv(csv_path, low_memory=False)
    df_price['block_timestamp'] = pd.to_datetime(df_price['block_timestamp'])

    # start drawing!!!
    outRange = 0
    for name in tqdm(df_info.index, desc='Processing Files'):

        name_m = df_info.loc[name, 'name']

        path = save_path + '/price_result/' +name+ '.csv'
        path_result = os.path.join(data_folder_path, name + '.csv')
        # path_result = '01_init/'+receipt+"."+name+'.result.csv' # 读取demeter结果数据
        # ,index,id,lower_tick,upper_tick,tx_type,block_number,tx_hash,log_index,blk_time,liquidity,final_amount0,final_amount1,tick_id,price_upper,price_lower
        # 26,28,715592,201700,203330,BURN,40207453,0x0f598614e22f61209a80200a579e9483a4e53e10dfd8343168cf7b06769ee529,373,2023-03-11 04:09:41,-180180745166465.0,185728532.0,2.4438321523982784e+17,201700-203330-0,1740.6772640717006,1478.8761174167462
        action = pd.read_csv(path, sep=',', index_col=[], dtype=object, header=0)
        # 删掉collect数据
        action = action[action.tx_type != 'COLLECT']
        action.loc[:, "blk_time"] = action.loc[:, "blk_time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        # print(type(action.loc[:,'block_timestamp'][0]), action.loc[:,'block_timestamp'][0])
        action = action[action.blk_time >= datetime.strptime('2024-03-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
        action = action[action.blk_time <= datetime.strptime('2024-09-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
        if action.empty:
            outRange+=1
            print("time out of range!   "+str(outRange))
            continue

        action.loc[:, "blk_time"] = action.loc[:, "blk_time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))

        action = action.sort_values(['blk_time', 'tx_type'], ascending=[True, False])
        action.index = action.loc[:, "blk_time"]
        action.loc[:,'tick_id'] = action.loc[:,'lower_tick']+action.loc[:,'upper_tick']

        try:
            action[['price_lower', 'price_upper','liquidity']] = action[['price_lower', 'price_upper','liquidity']].astype(float)
        except:
            action[['lower_tick', 'upper_tick','liquidity']] = action[['lower_tick', 'upper_tick','liquidity']].astype(float)
            action[['lower_tick', 'upper_tick']] = action[['lower_tick', 'upper_tick']].astype('Int64')

        # 删掉burn liquidity=0的数据
        idx = action[(action.tx_type == 'BURN')&(action.liquidity == 0)].index.values
        action = action.drop(idx)


        result = pd.read_csv(path_result, sep=',', index_col=[], dtype=object, header=0)
        result.rename(columns={result.columns[0]: 'hour'}, inplace=True)
        # print(result.columns)
        try:
            result.loc[:, "hour"] = result.loc[:, "hour"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            result = result[result["hour"] >= datetime.strptime('2024-03-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
            result = result[result["hour"] <= datetime.strptime('2024-09-15 00:00:00', "%Y-%m-%d %H:%M:%S")]
        except:
            print("wrong time format!")
            continue

        fig,axs = plt.subplots(3,1,sharex=True, height_ratios=[0.7,0.15,0.15], figsize=(20,10))
        fig.subplots_adjust(hspace=0)
        fig.suptitle(name + ' action', size=30)
        axs[0].set_ylabel('price')

        eth_temp = df_price[(df_price['block_timestamp'] >= result['hour'].min()) & (df_price['block_timestamp'] <= result['hour'].max())]
        axs[0].plot(eth_temp['block_timestamp'], eth_temp['price'], label='ETH Price')
        axs[0].legend()

        action_id = action.groupby('tick_id')
        # print(action_id)
        for id, df in action_id:
            a = 0
            try:
                df_mint = df[df.tx_type == 'MINT']
                start_hour = df_mint['blk_time'][0]
                # print("mint: "+start_hour)
                a = 1
                price_up = df_mint['price_upper'][0]
                price_low = df_mint['price_lower'][0]
                # print(price_up)
                # print(price_low)
            except:
                start_hour = action.loc[:, 'blk_time'].min()

            try:
                df_mint = df[df.tx_type == 'BURN']
                end_hour = df_mint['blk_time'][-1]
                # print("burn: "+end_hour)
                if a == 0:
                    price_up = df_mint['price_upper'][-1]
                    price_low = df_mint['price_lower'][-1]
            except:
                end_hour = action.loc[:, 'blk_time'].max()
            dates = pd.date_range(start_hour, end_hour, freq='1T').to_list()
            num = len(dates)

            if price_up > 5000:
                up_array = np.repeat(5000,num)
                #print(name, id, price_up, "out of range")
            else:
                #print('repeating')
                up_array = np.repeat(price_up, num)
            low_array = np.repeat(price_low,num)

            #print(dates[0],dates[-1], price_up, price_low)
            if len(up_array) == 1:
                #print('1')
                axs[0].vlines(dates, low_array, up_array, linewidth=3, colors='r', alpha=1)
            elif (price_up - price_low) <= 2:
                #print('2')
                axs[0].hlines(np.array([price_up, price_low]), dates[0], dates[-1], linewidth=5, colors='r',alpha=1)
            else:
                axs[0].fill_between(dates, up_array, low_array,interpolate=True, step='mid', alpha=.25)

        # add other 
        # axs[2].plot(eth_temp['date'], eth_temp['seven_yearly_vol'])
        # axs[2].set_ylabel('eth fluctuation')
        # axs[2].set_xlabel('time')


        # try:
        #     start_day = result['hour'].iloc[0]
        # except Exception as e:
        #     print(os.path.splitext(os.path.basename(file_path))[0] + ": no start day!_____" + str(no_start_day))
        #     continue

        # return_index_start = float(result.loc[result['hour'] == start_day, 'cumulate_return_rate'].iloc[0])
        # result['cumulate_return_rate'] = result['cumulate_return_rate'].astype(float) / return_index_start


        # axs[1].plot(result['hour'], result['cumulate_return_rate'])
        # axs[1].set_ylabel('cumulate value')
        # axs[1].set_xlabel('time')
        # axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))


        plt.xticks(rotation=30)

        plt.savefig(save_path1+'/'+str(name_m) + ' _strategy.png', bbox_inches='tight')
        plt.close()
for pool in pool_list_10_lp:
    print(f"================== visuallize action for high alpha address: {pool} ==================")
    visuallize_action_plot(pool)