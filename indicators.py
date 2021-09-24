import time

import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi
from utils import Plot_OHCL
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from cointanalysis import CointAnalysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

def GenerateGAF(all_ts, window_size, rolling_length, fname, normalize_window_scaling=1.0, method='summation',
                scale='[0,1]'):
    # 取得時間序列長度
    n = len(all_ts)

    # 我們要避免微觀被過度放大, 所以移動的 window_size 是原本的 normalize_window_scaling 倍
    moving_window_size = int(window_size * normalize_window_scaling)

    # 根據我們的滾動大小，將資料切成一組一組
    n_rolling_data = int(np.floor((n - moving_window_size) / rolling_length))

    # 最終的 GAF
    gramian_field = []

    # 紀錄價格，用來畫圖
    # Prices = []

    # 開始從第一筆資料前進
    for i_rolling_data in trange(n_rolling_data, desc="Generating...", ascii=True):

        # 起始位置
        start_flag = i_rolling_data * rolling_length
        # 整個窗格的資料先從輸入時間序列中取出來
        full_window_data = list(all_ts[start_flag: start_flag + moving_window_size])

        # 紀錄窗格的資料，用來畫圖
        # Prices.append(full_window_data[-int(window_size*(normalize_window_scaling-1)):])

        # 因為等等要做cos/sin運算, 所以先標準化時間序列
        rescaled_ts = np.zeros((moving_window_size, moving_window_size), float)
        min_ts, max_ts = np.min(full_window_data), np.max(full_window_data)
        if scale == '[0,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (full_window_data - min_ts) / diff
        if scale == '[-1,1]':
            diff = max_ts - min_ts
            if diff != 0:
                rescaled_ts = (2 * full_window_data - diff) / diff

        # 留下原始 window_size 長度的資料
        rescaled_ts = rescaled_ts[-int(window_size * (normalize_window_scaling - 1)):]

        # 計算 Gramian Angular Matrix
        this_gam = np.zeros((window_size, window_size), float)
        sin_ts = np.sqrt(np.clip(1 - rescaled_ts ** 2, 0, 1))
        if method == 'summation':
            # cos(x1+x2) = cos(x1)cos(x2) - sin(x1)sin(x2)
            this_gam = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)
        if method == 'difference':
            # sin(x1-x2) = sin(x1)cos(x2) - cos(x1)sin(x2)
            this_gam = np.outer(sin_ts, rescaled_ts) - np.outer(rescaled_ts, sin_ts)

        gramian_field.append(this_gam)

        # 清理記憶體占用
        del this_gam

    # 輸出 Gramian Angular Field
    np.array(gramian_field).dump('%s_gaf.pkl' % fname)

    # 清理記憶體占用
    del gramian_field


def PlotHeatmap(all_img, save_dir='output_img'):
    # 建立輸出資料夾
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 取得資料總長度
    total_length = all_img.shape[0]

    # 計算輸出圖片名稱的補零數量
    fname_zero_padding_size = int(np.ceil(np.log10(total_length)))

    # 輸出圖片
    for img_no in trange(total_length, desc="Output Heatmaps...", ascii=True):
        this_fname = str(img_no).zfill(fname_zero_padding_size)
        plt.imshow(all_img[img_no], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.savefig("%s/%s.png" % (save_dir, this_fname), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()

def AddIndicators(df,df2):
    #Adding cointegration factors
    df = df.reset_index()
    df2 = df2.reset_index()
    GenerateGAF(all_ts= df['Close'],
                window_size=50,
                rolling_length=1,
                fname="image_data")
    gafTFUEL_close = np.load('%s_gaf.pkl' % "image_data", allow_pickle=True)
    GenerateGAF(all_ts=df2['Close'],
                window_size=50,
                rolling_length=1,
                fname="image_dataBTC")
    gafBTC_close = np.load('%s_gaf.pkl' % "image_dataBTC", allow_pickle=True)
    GenerateGAF(all_ts=df['Volume'],
                window_size=50,
                rolling_length=1,
                fname="image_dataVOL")
    gafTFUEL_volume = np.load('%s_gaf.pkl' % "image_dataVOL", allow_pickle=True)
    GenerateGAF(all_ts=df2['Volume'],
                window_size=50,
                rolling_length=1,
                fname="image_dataVOLBTC")
    gafBTC_volume = np.load('%s_gaf.pkl' % "image_dataVOLBTC", allow_pickle=True)
    training_data = []
    for i, image in enumerate(gafTFUEL_volume):
        new_image_vol = np.reshape([image, gafBTC_volume[i],gafTFUEL_close[i],gafBTC_close[i]], (4, 50, 50))
        training_data.append([new_image_vol])
    return(training_data)
    # df2 = df2.reset_index()
    # i = 30
    # cointl1 = []
    # coeff1 = []
    # std1 = []
    # j = 0
    # while j < 29:
    #     cointl1.append(0)
    #     coeff1.append(0)
    #     std1.append(0)
    #     j = j + 1
    # while i < len(df) + 1:
    #         dfs = df[i - 30:i]['Close'].values
    #         dfs2 = df2[i - 30:i]['Close'].values
    #         X1 = np.array([dfs, dfs2]).T
    #         coint1 = CointAnalysis().fit(X1)
    #         coeff1.append(coint1.coef_[0])
    #         spread1 = coint1.transform(X1)
    #         cointl1.append(spread1[-1])
    #         std1.append(coint1.std_)
    #         i = i + 1
    # df['cointl'] = cointl1
    # df['coeff'] = coeff1
    # df['std'] = std1
    # df['Close2'] = df2['Close']
    # # Add Simple Moving Average (SMA) indicators
    # df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    # df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    # df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    # #df["sma300"] = SMAIndicator(close=df["Close"], window=300, fillna=True).sma_indicator()
    # # Add Bollinger Bands indicator
    # indicator_bb = BollingerBands(close=df["Close"], window=25, window_dev=2)
    # df['bb_bbm'] = indicator_bb.bollinger_mavg()
    # df['bb_bbh'] = indicator_bb.bollinger_hband()
    # df['bb_bbl'] = indicator_bb.bollinger_lband()
    #
    # # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    # indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    # df['psar'] = indicator_psar.psar()
    #
    # # Add Moving Average Convergence Divergence (MACD) indicator
    # df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas
    #
    # # Add Relative Strength Index (RSI) indicator
    # df["RSI"] = rsi(close=df["Close"], window=20, fillna=True) # mazas
    # df["sma72"] = SMAIndicator(close=df2["Close"], window=7, fillna=True).sma_indicator()
    # df["sma252"] = SMAIndicator(close=df2["Close"], window=25, fillna=True).sma_indicator()
    # df["sma992"] = SMAIndicator(close=df2["Close"], window=99, fillna=True).sma_indicator()
    # #df["sma300"] = SMAIndicator(close=df["Close"], window=300, fillna=True).sma_indicator()
    # # Add Bollinger Bands indicator
    # indicator_bb2 = BollingerBands(close=df2["Close"], window=25, window_dev=2)
    # df['bb_bbm2'] = indicator_bb2.bollinger_mavg()
    # df['bb_bbh2'] = indicator_bb2.bollinger_hband()
    # df['bb_bbl2'] = indicator_bb2.bollinger_lband()
    #
    # # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    # indicator_psar2 = PSARIndicator(high=df2["High"], low=df2["Low"], close=df2["Close"], step=0.02, max_step=2, fillna=True)
    # df['psar'] = indicator_psar2.psar()
    #
    # # Add Moving Average Convergence Divergence (MACD) indicator
    # df["MACD"] = macd(close=df2["Close"], window_slow=26, window_fast=12, fillna=True) # mazas
    #
    # # Add Relative Strength Index (RSI) indicator
    # df["RSI"] = rsi(close=df2["Close"], window=30, fillna=True) # mazas
    # df = df.loc[:, df.columns != 'Date']
    # df = df.loc[:, df.columns != 'index']
    return df

def DropCorrelatedFeatures(df, threshold, plot):
    df_copy = df.copy()

    # Remove OHCL columns
    df_drop = df_copy.drop(["Date", "Open", "High", "Low", "Close", "Volume"], axis=1)

    # Calculate Pierson correlation
    df_corr = df_drop.corr()

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold or df_corr.iloc[i,j] <= -threshold:
                if columns[j]:
                    columns[j] = False
                    
    selected_columns = df_drop.columns[columns]

    df_dropped = df_drop[selected_columns]

    if plot:
        # Plot Heatmap Correlation
        fig = plt.figure(figsize=(8,8))
        ax = sns.heatmap(df_dropped.corr(), annot=True, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        fig.tight_layout()
        plt.show()
    
    return df_dropped

def get_trend_indicators(df, threshold=0.5, plot=False):
    df_trend = df.copy()
    
    # add custom trend indicators
    df_trend["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df_trend["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df_trend["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()

    df_trend = add_trend_ta(df_trend, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_trend, threshold, plot)

def get_volatility_indicators(df, threshold=0.5, plot=False):
    df_volatility = df.copy()
    
    # add custom volatility indicators
    # ...

    df_volatility = add_volatility_ta(df_volatility, high="High", low="Low", close="Close")

    return DropCorrelatedFeatures(df_volatility, threshold, plot)

def get_volume_indicators(df, threshold=0.5, plot=False):
    df_volume = df.copy()
    
    # add custom volume indicators
    # ...

    df_volume = add_volume_ta(df_volume, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_volume, threshold, plot)

def get_momentum_indicators(df, threshold=0.5, plot=False):
    df_momentum = df.copy()
    
    # add custom momentum indicators
    # ...

    df_momentum = add_momentum_ta(df_momentum, high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_momentum, threshold, plot)

def get_others_indicators(df, threshold=0.5, plot=False):
    df_others = df.copy()
    
    # add custom indicators
    # ...

    df_others = add_others_ta(df_others, close="Close")

    return DropCorrelatedFeatures(df_others, threshold, plot)

def get_all_indicators(df, threshold=0.5, plot=False):
    df_all = df.copy()
    
    # add custom indicators
    # ...

    df_all = add_all_ta_features(df_all, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return DropCorrelatedFeatures(df_all, threshold, plot)

def indicators_dataframe(df, threshold=0.5, plot=False):
    trend       = get_trend_indicators(df, threshold=threshold, plot=plot)
    volatility  = get_volatility_indicators(df, threshold=threshold, plot=plot)
    volume      = get_volume_indicators(df, threshold=threshold, plot=plot)
    momentum    = get_momentum_indicators(df, threshold=threshold, plot=plot)
    others      = get_others_indicators(df, threshold=threshold, plot=plot)
    #all_ind = get_all_indicators(df, threshold=threshold)

    final_df = [df, trend, volatility, volume, momentum, others]
    result = pd.concat(final_df, axis=1)

    return result

if __name__ == "__main__":   
    df = pd.read_feather('./TFUEL15m.feather')
    #df = df.sort_values('Date')
    #df = AddIndicators(df)

    #test_df = df[-400:]

    #Plot_OHCL(df)
    get_others_indicators(df, threshold=0.5, plot=True)
