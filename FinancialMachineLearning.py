# ===============================================================================================================
#           Libraries
# =================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import re
import os
import time
from collections import OrderedDict as od
import math
import sys
import datetime as dt

from pathlib import PurePath, Path
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

import scipy.stats as stats
from scipy import interp

import copyreg, types, multiprocessing as mp
import copy
import platform
from multiprocessing import cpu_count

from numba import jit

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm, tqdm_notebook

import warnings

#statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import log_loss, accuracy_score
from itertools import product
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection._split import _BaseKFold
from sklearn import metrics

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
pd.set_option('display.max_rows', 100)
pbar = ProgressBar()
pbar.register()

# ===============================================================================================================
#           Bar Sampling
# =================================================================================================================

def getDataFrame(df):
    """
    High Frequency Data를 정리해주는 함수입니다.

    Return
    ----------------------------
    - pandas.DataFrame형태의 OHCL Data가 반환됩니다
    """
    temp = df[['price', 'buy', 'sell', 'volume']]
    temp['v'] = temp.volume
    temp['dv'] = temp.volume * temp.price
    temp.index = pd.to_datetime(temp.index)
    return temp

@jit(nopython = True)
def numba_isclose(a, b, rel_tol = 1e-09, abs_tol = 0.0):
    # rel_tol: relative tolerance
    # abs_tol: absolute tolerance
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a),np.fabs(b)),abs_tol)

def getOHLC(ref, sub):
    """
    fn: get ohlc from custom

    # args
        ref: reference pandas series with all prices
        sub: custom tick pandas series

    # returns
        tick_df: dataframe with ohlc values
    """
    ohlc = []
    # for i in tqdm(range(sub.index.shape[0]-1)):
    for i in range(sub.index.shape[0] - 1):
        start, end = sub.index[i], sub.index[i + 1]
        tmp_ref = ref.loc[start:end]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        o, h, l, c = sub.iloc[i], max_px, min_px, sub.iloc[i + 1]
        ohlc.append((end, start, o, h, l, c))
    cols = ['end', 'start', 'open', 'high', 'low', 'close']
    return (pd.DataFrame(ohlc, columns = cols))

@jit(nopython=True)
def madOutlier(y, thresh = 3.):
    """
    outlier를 탐지하는 함수입니다.
    :param y: pandas.Series 형태의 Price 계열 input data입니다
    :param thresh: outlier를 탐지하기 위한 구간을 지정합니다(default = 3.0)
    :return:
    """
    median = np.median(y)
    print(median)
    diff = np.sum((y - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    print(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    print(modified_z_score)
    return modified_z_score > thresh

def BarSampling(df, column, threshold, tick = False):
    """
    Argument
    ----------------------------
    df : getDataFrame 함수의 output을 입력으로 사용합니다
    column : 기준으로 사용할 column을 지정합니다
        'price' - time bar 사용
        'v' - volume 사용
        'dv' - dollar value 사용
    tick : tick bar를 사용하고 싶은 경우 True로 변경합니다

    Hyperparameter
    ----------------------------
    threshold : threshold를 넘길 때마다 Sampling

    output
    ----------------------------
    DataFrame 형태의 Sampling된 Data가 반환됩니다

    """
    t = df[column]
    ts = 0
    idx = []
    if tick:
        for i, x in enumerate(t):
            ts += 1
            if ts >= threshold:
                idx.append(i)
                ts = 0
    else:
        for i, x in enumerate(t):
            ts += x
            if ts >= threshold:
                idx.append(i)
                ts = 0
    return df.iloc[idx].drop_duplicates()


def get_ratio(df, column, n_ticks):
    """
    Argument
    ----------------------------
    df : bar_sampling 함수의 output을 입력으로 사용합니다
    column : 비율 기준 설정 (dollar_value, volume)
    n_ticks : tick 지정

    """
    return df[column].sum() / n_ticks


def select_sample_data(ref, sub, price_col, date):
    """
    DatetimeIndex를 index로 가진 Data를 기반으로 Sample Data를 선정합니다

    # args
        ref: 틱을 가지고 있는 DataFrame
        sub: subordinated pd.DataFrame of prices
        price_col: str(), price colume
        date: str(), date to select

    # returns
        xdf: ref pd.Series
        xtdf: subordinated pd.Series
    """

    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]

    return xdf, xtdf

def count_bars(df, price_col = 'price'):
    """
    일주일에 Bar가 Sampling되는 횟수를 계산해 줍니다
    """
    return df.groupby(pd.Grouper(freq='1W'))[price_col].count()

def scale(s):
    """
    비교를 위해 Scale을 조정해 줍니다
    """
    return (s - s.min()) / (s.max() - s.min())

def getReturns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def get_test_stats(bar_types, bar_returns, test_func, *args, **kwds):
    dct = {bar: (int(bar_ret.shape[0]), test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0: 'sample size', 1: f'{test_func.__name__}_stat'}).T)
    return df

def df_rolling_autocorr(df, window, lag = 1):
    """
    DataFrame의 rolling column-wise autocorrelation을 계산합니다
    """
    return (df.rolling(window = window).corr(df.shift(lag)))

def signed_tick(tick, initial_value=1.0):
    diff = tick['price'] - tick['price'].shift(1)
    return (abs(diff) / diff).ffill().fillna(initial_value)

def tick_imbalance_bar(tick, initial_expected_bar_size = 150, initial_expected_signed_tick = .1,
                       lambda_bar_size = .1, lambda_signed_tick = .1):
    tick = tick.sort_index(ascending = True)
    tick = tick.reset_index()

    # Part 1. Tick imbalance 값을 기반으로, bar numbering(`tick_imbalance_group`)
    tick_imbalance = signed_tick(tick).cumsum().values
    tick_imbalance_group = []

    expected_bar_size = initial_expected_bar_size
    expected_signed_tick = initial_expected_signed_tick
    expected_tick_imbalance = expected_bar_size * expected_signed_tick

    current_group = 1
    previous_i = 0

    for i in range(len(tick)):
        tick_imbalance_group.append(current_group)

        if abs(tick_imbalance[i]) >= abs(expected_tick_imbalance):  # 수식이 복잡해 보이지만 EMA임.
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            expected_signed_tick = (lambda_signed_tick * tick_imbalance[i] /
                                    (i - previous_i + 1) + (1 - lambda_signed_tick) * expected_signed_tick)
            expected_tick_imbalance = expected_bar_size * expected_signed_tick

            tick_imbalance -= tick_imbalance[i]

            previous_i = i
            current_group += 1

    # Part 2. Bar numbering 기반으로, OHLCV bar 생성
    tick['tick_imbalance_group'] = tick_imbalance_group
    groupby = tick.groupby('tick_imbalance_group')

    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()

    bars.set_index('t', inplace=True)

    return bars

def tick_runs_bar(tick, initial_expected_bar_size, initial_buy_prob,
                  lambda_bar_size=.1, lambda_buy_prob=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_tick_sell = _signed_tick.apply(lambda v: -v if v < 0 else 0).cumsum()
    group = []
    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    expected_runs = expected_bar_size * max(buy_prob, 1 - buy_prob)
    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_tick_buy[i], imbalance_tick_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_tick_sell -= imbalance_tick_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

def volume_runs_bar(tick, initial_expected_bar_size, initial_buy_prob, initial_buy_volume,
                    initial_sell_volume, lambda_bar_size=.1, lambda_buy_prob=.1,
                    lambda_buy_volume=.1, lambda_sell_volume=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    _signed_volume = _signed_tick * tick['volume']
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_buy = _signed_volume.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_sell = _signed_volume.apply(lambda v: v if -v < 0 else 0).cumsum()

    group = []

    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    buy_volume = initial_buy_volume
    sell_volume = initial_sell_volume
    expected_runs = expected_bar_size * max(buy_prob * buy_volume, (1 - buy_prob) * sell_volume)

    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_volume_buy[i], imbalance_volume_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            buy_volume = (lambda_buy_volume * imbalance_volume_buy[i] + (1 - lambda_buy_volume) * buy_volume)
            sell_volume = (lambda_sell_volume * imbalance_volume_sell[i] + (1 - lambda_sell_volume) * sell_volume)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_volume_buy -= imbalance_volume_buy[i]
            imbalance_volume_sell -= imbalance_volume_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

def getRunBars(tick, initial_expected_bar_size, initial_buy_prob, initial_buy_volume, initial_sell_volume,
               ticker = 'volume', lambda_bar_size=.1, lambda_buy_prob=.1, lambda_buy_volume=.1, lambda_sell_volume=.1):
    tick = tick.sort_index(ascending=True)
    tick = tick.reset_index()
    _signed_tick = signed_tick(tick)
    _signed_volume = _signed_tick * tick[ticker]
    imbalance_tick_buy = _signed_tick.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_buy = _signed_volume.apply(lambda v: v if v > 0 else 0).cumsum()
    imbalance_volume_sell = _signed_volume.apply(lambda v: v if -v < 0 else 0).cumsum()

    group = []

    expected_bar_size = initial_expected_bar_size
    buy_prob = initial_buy_prob
    buy_volume = initial_buy_volume
    sell_volume = initial_sell_volume
    expected_runs = expected_bar_size * max(buy_prob * buy_volume, (1 - buy_prob) * sell_volume)

    current_group = 1
    previous_i = 0
    for i in range(len(tick)):
        group.append(current_group)

        if max(imbalance_volume_buy[i], imbalance_volume_sell[i]) >= expected_runs:
            expected_bar_size = (lambda_bar_size * (i - previous_i + 1) + (1 - lambda_bar_size) * expected_bar_size)
            buy_prob = (lambda_buy_prob * imbalance_tick_buy[i] /
                        (i - previous_i + 1) + (1 - lambda_buy_prob) * buy_prob)
            buy_volume = (lambda_buy_volume * imbalance_volume_buy[i] + (1 - lambda_buy_volume) * buy_volume)
            sell_volume = (lambda_sell_volume * imbalance_volume_sell[i] + (1 - lambda_sell_volume) * sell_volume)
            previous_i = i
            imbalance_tick_buy -= imbalance_tick_buy[i]
            imbalance_volume_buy -= imbalance_volume_buy[i]
            imbalance_volume_sell -= imbalance_volume_sell[i]
            current_group += 1
    tick['group'] = group
    groupby = tick.groupby('group')
    bars = groupby['price'].ohlc()
    bars[ticker] = groupby[ticker].sum()
    bars['value'] = groupby['value'].sum()
    #bars[['volume', 'value']] = groupby[['volume', 'value']].sum()
    bars['t'] = groupby['t'].first()
    bars.set_index('t', inplace=True)
    return bars

@jit(nopython=True)
def getSequence(p0,p1,bs):
    if numba_isclose((p1-p0),0.0,abs_tol=0.001):
        return bs[-1]
    else: return np.abs(p1-p0)/(p1-p0)

@jit(nopython=True)
def getImbalance(t):
    """Noted that this function return a list start from the 2nd obs"""
    bs = np.zeros_like(t)
    for i in np.arange(1,bs.shape[0]):
        bs[i-1] = getSequence(t[i-1],t[i],bs[:i-1])
    return bs[:-1] # remove the last value

def test_t_abs(absTheta, t, E_bs):
    """
    Bool function to test inequality
    * row is assumed to come from df.itertuples()
    - absTheta: float(), row.absTheta
    - t: pd.Timestamp
    - E_bs: float, row.E_bs
    """
    return (absTheta >= t * E_bs)

def getAggImalanceBar(df):
    """
    Implements the accumulation logic
    """
    start = df.index[0]
    bars = []
    for row in df.itertuples():
        t_abs = row.absTheta
        rowIdx = row.Index
        E_bs = row.E_bs

        t = df.loc[start:rowIdx].shape[0]
        if t < 1: t = 1  # if t less than 1, set equal to 1
        if test_t_abs(t_abs, t, E_bs):
            bars.append((start, rowIdx, t))
            start = rowIdx
    return bars

def getRolledSeries(series, dictio):
    gaps = rollGaps(series, dictio)
    for field in ['Close', 'Volume']:
        series[field] -= gaps
    return series

def rollGaps(series, dictio, matchEnd=True):
    # Compute gaps at each roll, between previous close and next open
    rollDates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps = series[dictio['Close']] * 0
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in rollDates] # index of days prior to roll
    gaps.loc[rollDates[1:]] = series[dictio['Open']].loc[rollDates[1:]] - series[dictio['Close']].iloc[iloc[1:]].values
    gaps = gaps.cumsum()
    if matchEnd:
        gaps -= gaps.iloc[-1]
    return gaps

def getBollingerRange(data: pd.Series, window: int = 21, width: float = 0.005):
    """
    Bollinger Band를 구축하는 Parameter를 return으로 하는 함수입니다
    :param data: pandas.Series 형태의 price Data를 input으로 합니다
    :param window: Rolling할 기간을 지정하는 Hyper Parameter입니다
    :param width:
    :return:
    """
    avg = data.ewm(span = window).mean()
    std0 = avg * width
    lower = avg - std0
    upper = avg + std0

    return avg, upper, lower, std0

def pcaWeights(cov, riskDist = None, risktarget = 1.0, valid = False):
    """
    Rick Allocation Distribution을 따라서 Risk Target을 매치합니다
    :param cov: pandas.DataFrame 형태의 Covariance Matrix를 input으로 합니다
    :param riskDist: 사용자 지정 리스크 분포입니다. None이라면 코드는 모든 리스크가 최소 고유값을 갖는 주성분에 배분되는 것으로 가정합니다.
    :param risktarget: riskDist에서의 비중을 조절할 수 있습니다. 기본값은 1.0입니다
    :param valid: riskDist를 검증하고 싶으면 True로 지정합니다. 이 경우 결과값은 (wghts, ctr)의 형태로 출력됩니다
    :return:
    """
    eVal, eVec = np.linalg.eigh(cov)  # Hermitian Matrix
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1.

    loads = riskTarget * (riskDist / eVal) ** 0.5
    wghts = np.dot(eVec, np.reshape(loads, (-1, 1)))

    if vaild == True:
        ctr = (loads / riskTarget) ** 2 * eVal  # riskDist 검증
        return (wghts, ctr)
    else:
        return wghts

def CusumEvents(df: pd.Series, limit: float):
    """
    이벤트 기반의 표본 추출을 하는 함수입니다
    :param df: pandas.Series 형태의 가격 데이터입니다
    :param limit: Barrier를 지정하는 threshold입니다. numerical data이며, 높게 지정할 수록 label이 적게 추출됩니다
    :return:
    """
    idx, _up, _dn = [], 0, 0
    diff = df.diff()
    for i in range(len(diff)):
        if _up + diff.iloc[i] > 0:
            _up = _up + diff.iloc[i]
        else:
            _up = 0

        if _dn + diff.iloc[i] < 0:
            _dn = _dn + diff.iloc[i]
        else:
            _dn = 0

        if _up > limit:
            _up = 0;
            idx.append(i)
        elif _dn < - limit:
            _dn = 0;
            idx.append(i)
    return idx

# ===============================================================================================================
#           Labeling
# =================================================================================================================

def getDailyVolatility(close, span = 100):
    """
    Daily Rolling Volatility를 추정하는 함수입니다

    Argument
    ----------------------------
    span(default = 100) : Rolling할 Number of Days를 지정
    """
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1],
                     index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span = span).std().rename('dailyVol')
    return df0

def addVerticalBarrier(tEvents, close, numDays=1):
    """
    Position Holding 기간을 지정하여 Vertical Barrier를 구축합니다

    Argument
    ----------------------------
    tEvents : getTEvents 함수의 output
    close : pandas Series 형태인 가격에 관한 Data

    Hyper Parameter
    ----------------------------
    numDays (default = 1) : 어느정도의 기간을 Rolling할 것인지 지정

    """
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def tradableHour(i, start='09:40', end='15:50'):
    """
    : param i: a datetimeIndex value
    : param start: the start of the trading hour
    : param end: the end of the trading hour

    : return: bool, is tradable hour or not"""
    time = i.strftime('%H:%M')
    return (time < end and time > start)

def getTEvents(gRaw, h, symmetric=True, isReturn=False):
    """
    Symmetric CUSUM Filter
    Sample a bar t iff S_t >= h at which point S_t is reset
    Multiple events are not triggered by gRaw hovering around a threshold level
    It will require a full run of length h for gRaw to trigger an event

    Two arguments:
        gRaw: the raw time series we wish to filter (gRaw), e.g. return
        h: threshold

    Return:
        pd.DatatimeIndex.append(tEvents):
    """
    tEvents = []
    if isReturn:
        diff = gRaw
    else:
        diff = gRaw.diff()
    if symmetric:
        sPos, sNeg = 0, 0
        if np.shape(h) == ():

            for i in diff.index[1:]:
                sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
                if sNeg < -h and tradableHour(i):
                    sNeg = 0;
                    tEvents.append(i)
                elif sPos > h and tradableHour(i):
                    sPos = 0;
                    tEvents.append(i)
        else:
            for i in diff.index[1:]:
                sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
                if sNeg < -h[i] and tradableHour(i):
                    sNeg = 0;
                    tEvents.append(i)
                elif sPos > h[i] and tradableHour(i):
                    sPos = 0;
                    tEvents.append(i)
    else:
        sAbs = 0
        if np.shape(h) == ():

            for i in diff.index[1:]:
                sAbs = sAbs + diff.loc[i]
                if sAbs > h and tradableHour(i):
                    sAbs = 0;
                    tEvents.append(i)

        else:
            for i in diff.index[1:]:
                sAbs = sAbs + diff.loc[i]
                if sAbs > h[i] and tradableHour(i):
                    sAbs = 0;
                    tEvents.append(i)

    return pd.DatetimeIndex(tEvents)

def getTripleBarrier(close, events, ptSl, molecule):
    """
    Triple Barrier Method를 구현하는 함수입니다
    Horizonal Barrier, Vertical Barrier 중 어느 하나라도 Touch를 하면 Labeling을 진행합니다

    Argument
    ----------------------------
    close : Price 정보가 담겨 있는 pandas.Series 계열의 데이터를 input으로 넣습니다
    events : pandas.DataFrame으로서 다음의 열을 가집니다
        - t1 : Vertical Barrier의 Time Stamp 값입니다. 이 값이 np.nan이라면 Vertical Barrier가 없습니다
        - trgt : Horizonal Barrier의 단위 너비입니다

    ptSl : 음이 아는 두 실수값의 리스트입니다
        - ptSl[0] : trgt에 곱해서 Upper Barrier 너비를 설정하는 인수입니다. 값이 0이면 Upper Barrier가 존재하지 않습니다
        - ptSl[1] : trgt에 곱해서 Lower Barrier 너비를 설정하는 인수입니다. 값이 0이면 Lower Barrier가 존재하지 않습니다

    molecule : Single Thread에 의해 처리되는 Event Index의 부분 집합을 가진 리스트입니다

    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc: t1]  # 가격 경로
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # 수익률 경로
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # 가장 빠른 손절 시점
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # 가장 빠른 이익 실현 시점

    return out


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    베팅의 방향과 크기를 파악할 수 있는 함수입니다

    Argument
    ----------------------------
    close : Price 정보가 담겨 있는 pandas.Series 계열의 데이터를 input으로 넣습니다
    tEvents : 각 Triple Barrier Seed가 될 Time Stamp값을 가진 Pandas TimeIndex입니다
    ptSl : 음이 아는 두 실수값의 리스트로, 두 Barrier의 너비를 설정합니다
    trgt : 수익률의 절대값으로 표현한 목표 pandas.Series 객체의 데이터를 input으로 합니다향
    minRet : Triple Barrier 검색을 진행할 때 필요한 최소 목표 수익률입니다
    numThreads : 함수에서 현재 동시에 사용하고 있는 Thread의 수입니다

    t1(default = False) : Vertical Barrier의 Time Stamp를 가진 pandas.Series 객체의 데이터를 input으로 합니다
    side(default = None) : side 값을 input으로 넣습니다

    """
    # 1) 목표 구하기
    for i in tEvents:
        if i not in trgt.index:
            trgt[str(i)] = np.NaN
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet

    # 2) t1 구하기 (최대 보유 기간)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) t1에 손절을 적용해 이벤트 객체를 형성
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[side.index & trgt.index], ptSl[:2]

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis = 1).dropna(subset = ['trgt'])
    df0 = mpPandasObj(func = getTripleBarrier, pdObj=('molecule', events.index),
                          numThreads = numThreads, close = close, events = events, ptSl = np.array(ptSl_))
    events['t1'] = df0.dropna(how = 'all').min(axis = 1)  # pd.min ignores nan

    if side is None:
        events = events.drop('side', axis=1)

    return events

def getBins(events, close):
    """
    이벤트를 감지해 출력하는 함수입니다. 가능하다면 베팅 사이드에 대한 정보도 포함합니다.

    Argument
    ----------------------------
    events : 감지된 Events가 존재하는 pandas DataFrame형태의 input data입니다. 아래와 같은 column을 가집니다
        - t1 : event의 마지막 시간을 의미합니다
        - trgt : event의 Target을 의미합니다
        - side : Position의 방향을 의미합니다 (상승, 하락)

    - Case 1 ('side'가 이벤트에 없음) : bin in (-1, 1) 가격 변화에 의한 레이블
    - Case 2 ('side'가 이벤트에 있음) : bin in (0, 1) 손익(pnl)에 의한 레이블 (meta labeling)
    """

    # 1) 가격과 이벤트를 일치
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # 2) OUT 객체 생성
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling

    return out


def dropLabels(events, minPct=0.05):
    # 예제가 부족할 경우 가중치를 적용해 레이블을 제거한다.
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events



def getBinsNew(events, close, t1=None):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])

    if 'side' not in events_:
        # only applies when not meta-labeling
        # to update bin to 0 when vertical barrier is touched, we need the original
        # vertical barrier series since the events['t1'] is the time of first 
        # touch of any barrier and not the vertical barrier specifically. 
        # The index of the intersection of the vertical barrier values and the 
        # events['t1'] values indicate which bin labels needs to be turned to 0
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.

    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out

def get_up_cross(df):
    """
    이익 실현 구간을 지정하는 함수입니다
    """
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return df.fast[(crit1) & (crit2)]

def get_down_cross(df):
    """
    손실 한도 구간을 지정하는 함수입니다
    """
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return df.fast[(crit1) & (crit2)]

def getUpCross(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper.shift(1)
    crit2 = df[col] > df.upper
    return df[col][(crit1) & (crit2)]

def getDownCross(df, col):
    # col is price column
    crit1 = df[col].shift(1) > df.lower.shift(1)
    crit2 = df[col] < df.lower
    return df[col][(crit1) & (crit2)]

# ===============================================================================================================
#           Sample Weights
# =================================================================================================================

def getConcurrentBar(closeIdx, t1, molecule):
    """
    Bar별로 공존하는 Event의 개수를 계산하여 Label의 고유도를 계산하는 함수입니다
    :param closeIdx : 가격 계열의 data를 Value로 가지는 Index를 input으로 합니다
    :param t1 : pandas.Series형태의 데이터로, Vertical Barrier를 형성하는 timestamp 정보를 input으로 합니다
    :param molecule : 가중값이 계산될 이벤트의 시간 정보를 input으로 합니다. t1[molecule].max()이전에 발생하는 모든 이벤트는 개수에 영향을 미치게 됩니다
        molecule[0]은 가중값이 계산될 첫 이벤트 시간입니다
        molecule[-1]은 가중값이 계산될 마지막 이벤트 시간입니다
    :return count : pandas.Series 형태로 Concurrent Bar의 개수가 Bar마다 출력되어 나옵니다
    """
    # 1) [molecule[0], molecule[-1]]에서 Event를 탐색합니다
    # fill the unclosed events with the last available (index) date
    t1 = t1.fillna(closeIdx[-1]) # 드러난 이벤트들은 다른 가중값에 영향을 미쳐야 합니다
    t1 = t1[t1 >= molecule[0]] # molecule[0]의 마지막이나 이후에 발생하는 이벤트입니다
    # t1[molecule].max() 이전이나 시작 시에 발생하는 이벤트입니다
    t1 = t1.loc[: t1[molecule].max()]

    # 2) 바에서 발생하는 이벤트의 개수를 알아보는 과정입니다
    # find the indices begining start date ([t1.index[0]) and the furthest stop date (t1.max())
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    # form a 0-array, index: from the begining start date to the furthest stop date
    count = pd.Series(0, index=closeIdx[iloc[0]: iloc[1] + 1])
    # for each signal t1 (index: eventStart, value: eventEnd)
    for tIn, tOut in t1.iteritems():
        # add 1 if and only if [t_(i,0), t_(i.1)] overlaps with [t-1,t]
        count.loc[tIn: tOut] += 1  # every timestamp between tIn and tOut
    # compute the number of labels concurrents at t
    return count.loc[molecule[0]: t1[molecule].max()]  # only return the timespan of the molecule


def getAvgLabelUniq(t1, numCoEvents, molecule):
    """
    :param t1: pd series, timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
    :param numCoEvent: 
    :param molecule: the date of the event on which the weight will be computed
        + molecule[0] is the date of the first event on which the weight will be computed
        + molecule[-1] is the date of the last event on which the weight will be computed
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    # derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    # for each events
    for tIn, tOut in t1.loc[wght.index].iteritems():
        # tIn, starts of the events, tOut, ends of the events
        # the more the coEvents, the lower the weights
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn: tOut]).mean()
    return wght


def mpSampleWeights(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    out = events[['t1']].copy(deep=True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                                  t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'],
                                numCoEvents = numCoEvents)
    return out


def getBollingerBand(price, window = None, width = None, numsd = None):
    """
    Bollinger Band를 구축해주는 함수입니다
    :param price : pandas.Series 형태의 가격을 input으로 합니다
    :param window : rolling할 days의 값을 numerical input data로 지정합니다(default = 0)
    :param width : Bollinger Band 구축 시 상한 하한을 지정해주는 parameter입니다(default = 0)
    :param numsd : Bollinger Band 구축 시 상한 하한을 변동성으로 지정해주는 parameter입니다(default = 0)
    """
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1 + width)
        dnband = ave * (1 - width)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)
    if numsd:
        upband = ave + (sd * numsd)
        dnband = ave - (sd * numsd)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


def getSampleWeights(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    :return wght: pd.Series, the sample weight of each (volume) bar
    """
    out = events[['t1']].copy(deep=True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index, t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(mpSampleWeights, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents=numCoEvents)
    return out


def getIndMatrix(barIx, t1):
    """
    지표 행렬을 구축하는 함수입니다
    :param barIx: Bar의 index를 input으로 합니다
    :param t1: pandas.Series 형태의 Vertical Barrier의 timestamps를 input으로 합니다 (index: eventStart, value: eventEnd)
    :return indM: binary matrix, 각 관측치의 Label에 price Bar가 미치는 영향을 보여줍니다
    """
    indM = pd.DataFrame(0, index = barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):  # signal = obs
        indM.loc[t0: t1, i] = 1.  # each obs each column, you can see how many bars are related to an obs/
    return indM


def getAvgUniqueness(indM):
    """
    각 측성 관측값의 고유도(Uniuqeness) 평균을 반환합니다
    :param indM: getIndMatrix에 의해 구성된 Indicator Matrix를 input으로 합니다
    :return avgU: average uniqueness of each observed feature
    """
    # 지표 행렬로부터의 평균 고유도
    c = indM.sum(axis=1)  # concurrency, how many obs share the same bar
    u = indM.div(c, axis=0)  # uniqueness, the more obs share the same bar, the less important the bar is
    avgU = u[u > 0].mean()  # average uniquenessn
    return avgU


def seqBootstrap(indM, sLength=None):
    """
    Give the index of the features sampled by the sequential bootstrap
    :param indM: binary matrix, indicate what (price) bars influence the label for each observation
    :param sLength: optional, sample length, default: as many draws as rows in indM
    """
    # Generate a sample via sequential bootstrap
    if sLength is None:  # default
        sLength = indM.shape[1]  # sample length = # of rows in indM
    # Create an empty list to store the sequence of the draws
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()  # store the average uniqueness of the draw
        for i in indM:  # for every obs
            indM_ = indM[phi + [i]]  # add the obs to the existing bootstrapped sample
            # get the average uniqueness of the draw after adding to the new phi
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[
                -1]  # only the last is the obs concerned, others are not important
        prob = avgU / avgU.sum()  # cal prob <- normalise the average uniqueness
        phi += [np.random.choice(indM.columns, p=prob)]  # add a random sample from indM.columns with prob. = prob
    return phi


def main():
    # t0: t1.index; t1: t1.values
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
    # index of bars
    barIx = range(t1.max() + 1)
    # get indicator matrix
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi)
    print('Standard uniqueness:', getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:', getAvgUniqueness(indM[phi]).mean())

def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series()
    for _ in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def mainMC(numObs = 10, numBars = 100, maxH = 5, numIters = 1E6, numThreads = 24):
    # Monte Carlo experiments
    jobs = []
    for _ in range(int(numIters)):
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads = numThreads)
    print(pd.DataFrame(out).describe())
    return


def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn: tOut] / numCoEvents.loc[tIn: tOut]).sum()
    return wght.abs()


def SampleW(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    """
    out = events[['t1']].copy(deep=True)
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                              t1=events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()  # normalised, sum up to sample size

    return out

def getConcurUniqueness(close, events, numThreads):
    out = events[['t1']].copy(deep = True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx = close.index, t1 = out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents = numCoEvents)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents = numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()  # normalised, sum up to sample size
    return out

def getTimeDecay(tW, clfLastW = 1.):
    """
    apply piecewise-linear decay to observed uniqueness (tW)
    clfLastW = 1: no time decay
    0 <= clfLastW <= 1: weights decay linearly over time, but every obersevation still receives a strictly positive weight
    c = 0: weughts converge linearly to 0 as they become older
    c < 0: the oldest portion cT of the observations receive 0 weight
    c > 1: weights increase as they get older"""
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()  # cumulative sum of the observed uniqueness
    if clfLastW >= 0:  # if 0 <= clfLastW <= 1
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:  # if -1 <= clfLastW < 1
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0  # neg weight -> 0
    print(const, slope)
    return clfW

# ===============================================================================================================
#      Fractionally Differentiated Features
# =================================================================================================================

def getWeights(d, size):
    """
    thres > 0 유의미하지 않은 가중값을 제거하는 함수입니다
    :param d: 양의 실수인 미분 차수입니다. float data를 input으로 합니다
    :param size: 기간을 의미하는 parameter입니다. size가 클수록 긴 기간의 weight를 계산할 수 있습니다 (0으로 수렴합니다)
    :return:
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[:: -1]).reshape(-1, 1)
    return w

def fracDiff(series, d, thres = .01):
    """
    NaN을 처리하여 윈도우 너비를 증가시키는 함수입니다
    :param series: Price Sequence를 가지는 pandas.Series 형태의 data를 input으로 합니다
    :param d: 미분 차수를 Hyper Parameter로 넣습니다. float형 data를 input으로 합니다
    :param thres: 임계치를 설정하는 Hyper Parameter입니다. p-value가 0.01을 초과할 경우 계산이 정지됩니다
    :return:
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])  # each obs has a weight
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))  # cumulative weights
    w_ /= w_[-1]  # determine the relative weight-loss
    skip = w_[w_ > thres].shape[0]  # the no. of results where the weight-loss is beyond the acceptable value
    # 3) Apply weights to values
    df = {}  # empty dictionary
    for name in series.columns:
        # fill the na prices
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()  # create a pd series
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]  # find the iloc th obs

            test_val = series.loc[loc, name]  # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()

            if not np.isfinite(test_val).any():
                continue  # exclude NAs
            try:  # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
                df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def getWeights_FFD(d, thres):
    # thres>0 drops insignificant weights
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[:: -1]).reshape(-1, 1)[1:]
    return w

def fracDiff_FFD(series, d, thres=1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    # w = getWeights(d, series.shape[0])
    # w=getWeights_FFD(d,thres)
    width = len(w) - 1
    # 2) Apply weights to values
    df = {}  # empty dict
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()  # empty pd.series
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            # try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]
            # except:
            #     continue

        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def getOptimalFFD(data, start = 0, end = 1, interval = 10, t = 1e-5):
    """

    :param data:
    :param start:
    :param end:
    :param interval:
    :param t:
    :return:
    """
    d = np.linspace(start, end, interval)
    out = mpJobList(mpGetOptimalFFD, ('molecules', d), redux=pd.DataFrame.append, data=data)

    return out

def mpGetOptimalFFD(data, molecules, t=1e-5):
    cols = ['adfStat', 'pVal', 'lags', 'nObs', '95% conf']
    out = pd.DataFrame(columns=cols)

    for d in molecules:
        try:
            dfx = fracDiff_FFD(data.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx['price'], maxlag=1, regression='c', autolag=None)
            out.loc[d] = list(dfx[:4]) + [dfx[4]['5%']]
        except Exception as e:
            print(f'{d} error: {e}')
    return out

def OptimalFFD(data, start=0, end=1, interval=10, t=1e-5):
    for d in np.linspace(start, end, interval):
        dfx = fracDiff_FFD(data.to_frame(), d, thres=t)
        if sm.tsa.stattools.adfuller(dfx['price'], maxlag=1, regression='c', autolag=None)[1] < 0.05:
            return d
    print('no optimal d')
    return d

# =================================================================================================================
#           Cross Validation
# =================================================================================================================
def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the times of the training observations.
    Purge from the training set all observations whose labels overlapped in time with those labels included in the testing set
    :params t1: event timestamps
        —t1.index: Time when the observation started.
        —t1.value: Time when the observation ended.
    :params testTimes: pd.series, Times of testing observations.
    :return trn: pd.df, purged training set
    """
    # copy t1 to trn
    trn = t1.copy(deep=True)
    # for every times of testing obervation
    for i, j in testTimes.iteritems():
        # cond 1: train starts within test
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # cond 2: train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index
        # cond 3: train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        # drop the data that satisfy cond 1 & 2 & 3
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def getEmbargoTimes(times, pctEmbargo):
     """ Not sure if it works
     # Get embargo time for each bar
     :params times: time bars
     :params pctEmbargo: float, % of the bars will be embargoed
     :return trn: pd.df, purged training set
     """
     # cal no. of steps from the test data
     step = int(times.shape[0] * pctEmbargo)
     if step == 0:
         # if no embargo, the same data set
         mbrg=pd.Series(times,index=times)
     else:
         #
         mbrg=pd.Series(times[step:],index=times[:-step])
         mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
     return mbrg

class PurgedKFold(_BaseKFold):
    """
    K Fold Class를 확장하여 구간에 걸쳐 있는 레이블을 조정합니다.
    Train Data Set에서 Test Label 구간과 중첩 상태에 있는 관측값을 제거합니다.
    Test Set이 그 사이에 Train data의 Sample이 없이 연접해 있다고 가정합니다 (shuffle = False)
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            # if t1 is not a pd.series, raise error
            raise ValueError('Label Through Dates must be a pd.Series')
        # inherit _BaseKFold, no shuffle
        # Might be python 2x style
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1  # specify the vertical barrier
        self.pctEmbargo = pctEmbargo  # specify the embargo parameter (% of the bars)

    def split(self, X, y=None, groups=None):
        """
        :param X: the regressors, features
        :param y: the regressands, labels
        :param groups: None

        : return
            + train_indices: generator, the indices of training dataset
            + test_indices: generator, the indices of the testing dataset
        """
        if (pd.DataFrame(X).index == self.t1.index).sum() != len(self.t1):
            # X's index does not match t1's index, raise error
            raise ValueError('X and ThruDateValues must have the same index')
        # create an array from 0 to (X.shape[0]-1)
        indices = np.arange(X.shape[0])
        # the size of the embargo
        mbrg = int(X.shape[0] * self.pctEmbargo)
        # list comprehension, find the (first date, the last date + 1) of each split
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:  # for each split
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i: j]  # test indices are all the indices from i to j
            maxT1Idx = self.t1.index.searchsorted(
                self.t1[test_indices].max())  # find the max(furthest) vertical barrier among the test dates
            # index.searchsorted: find indices where element should be inserted (behind) to maintain the order
            # find all t1.indices (the start dates of the event) when t1.value (end date) < t0
            # i.e the left side of the training data
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                # indices[maxT1Idx+mbrg:]: the indices that is after the (maxTestDate + embargo period) [right training set]
                # concat the left training indices and the right training indices
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
        # the function return generators for the indices of training dataset and the indices of the testing dataset respectively
        yield train_indices, test_indices

def cvScore(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None, pctEmbargo=0):
    """
    Address two sklearn bugs
    1) Scoring functions do not know classes_
    2) cross_val_score will give different results because it weights to the fit method, but not to the log_loss method

    :params pctEmbargo: float, % of the bars will be embargoed
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        # if not using 'neg_log_loss' or 'accuracy' to score, raise error
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score  # import log_loss and accuracy_score
    if cvGen is None:  # if there is no predetermined splits of the test sets and the training sets
        # use the PurgedKFold to generate splits of the test sets and the training sets
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged
    score = []  # store the CV scores
    # for each fold
    for train, test in cvGen.split(X = X):
        # fit the model
        fit = clf.fit(X = pd.DataFrame(X).iloc[train, :], y = pd.DataFrame(y).iloc[train],
                      sample_weight = pd.DataFrame(sample_weight).iloc[train].values.reshape(1,-1)[0])
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(pd.DataFrame(X).iloc[test, :])  # predict the probabily
            # neg log loss to evaluate the score
            score_ = -1 * log_loss(pd.DataFrame(y).iloc[test], prob,
                                   sample_weight = pd.DataFrame(sample_weight).iloc[test].values.reshape(1,-1)[0],
                                   labels=clf.classes_)
        else:
            pred = fit.predict(pd.DataFrame(X).iloc[test, :])  # predict the label
            # predict the accuracy score
            score_ = accuracy_score(pd.DataFrame(y).iloc[test], pred,
                                    sample_weight = pd.DataFrame(sample_weight).iloc[test].values.reshape(1, -1)[0])
        score.append(score_)
    return np.array(score)

def crossValPlot(skf,classifier,X_,y_):
    """Code adapted from:

    """

    X = pd.DataFrame(X_)
    X = np.asarray(X)
    y = np.asarray(y_)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    f,ax = plt.subplots(figsize=(10,7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1,1))

# =================================================================================================================
#      Feature Importance
# =================================================================================================================

def featImpMDI(fit, featNames):
    """
    feat importance based on In-Sample (IS) mean impurity reduction
    :params fit: fit model
    :params featNames: str, the name of the features
    :return imp:
    """
    # find the feature importance of each estimator
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    # convert df from dicts, the keys are rows
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    # the column names are the feature names
    df0.columns = featNames
    # make sure features with 0 importance are not averaged
    # since the only reason for a 0 is that the feature is not randomly chosen
    df0 = df0.replace(0, np.nan)  # because max_features=1
    # imp is a dict that document ??
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)
    # feature importances add up to 1 and each is bounded between 0 and 1
    # but it is strange
    imp /= imp['mean'].sum()
    return imp


def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    """
    feature importance based on OOS score reduction
    1) Fit a classifier
    2) derive its OOS performance (neg_log_loss or accuracy)
    3) permutate each column of the featurs matrix (X), one at a time
    4) Derive the OOS performance after the permutation
    5) the worse the performance after, the more important the feature
    :params clf:
    :params X:
    :params y:
    :params cv:
    :params sample_weight:
    :params t1:
    :params pctEmbargo:
    :params scoring:

    :return
        imp
        scr0.mean()
    """
    # if the CV scoring scheme is neither 'neg_log_loss' or 'accuracy'
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')  # raise error/exception

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv with embargo

    scr0 = pd.Series()  # empty pd.series
    scr1 = pd.DataFrame(columns=X.columns)  # empty pd.df, but columns are the feature names (X,columns)

    for i, (train, test) in enumerate(cvGen.split(X=X)):  # for each CV fold
        # x0, y0, and w0 are the training features, training labels, and the sample weights for the training data respectively
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        # x1, y1, and w1 are the testing features, testing labels, and the sample weights for the testing data respectively
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        # fit the training set to the classifier
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':  # if the scoring is "neg_log_loss"
            prob = fit.predict_proba(X1)  # predict the prob
            scr0.loc[i] = -1 * log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)  # cal the neg_log_loss
        else:  # if the scoring is the "accuracy"
            pred = fit.predict(X1)  # predict label
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)  # produce the accuracy score
        for j in X.columns:  # for each feature
            X1_ = X1.copy(deep=True)  # X1_ = X1
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == 'neg_log_loss':  # if the scoring is "neg_log_loss"
                prob = fit.predict_proba(X1_)  # predict the prob
                scr1.loc[i, j] = -1 * log_loss(y1, prob, sample_weight=w1.values,
                                               labels=clf.classes_)  # cal the neg_log_loss
            else:  # if the scoring is the "accuracy"
                pred = fit.predict(X1_)  # predict label
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)  # produce the accuracy score
    # add -scr0 to scr1, axis to match Series index on 'index'
    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':  # if the scoring is "neg_log_loss"
        imp = imp / -scr1
    else:  # if the scoring is the "accuracy"
        imp = imp / (1. - scr1)
    # cal mean feature importance, std of mean importance
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    # return feature importance indicators and mean accuracy score
    return imp, scr0.mean()


def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    """
    Implementation of Single Feature Importance (SFI)
    Basically is to calculate all the cvScores of all the features
    :params featNames:
    :params clf:
    :params trnsX:
    :params cont:
    :params scoring:
    :params cvGen:

    :return imp:
    """
    # create an empty dataframe with column 'mean' and 'std'
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:  # for each feature  # import cvScore
        # cal the cvScores
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        # find the mean cvScores
        imp.loc[featName, 'mean'] = df0.mean()
        # find the std of cvScores
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0] ** -.5
    return imp


def get_eVec(dot, varThres):
    """
    compute eVec from dot prod matrix, reduce dimension
    :params dot:
    :params varThres:
    :return
        + eVal: eigen values
        + eVec: eigen vectors
    """
    # 1) compute eVec and eVal from dot
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    # eigen values are put into a pd.series, rank from most important to the least important
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    # eigen vectors are put into a pd.df, index = dot, columns = eVal
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    # ? in case there are additional columns, discard them all
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    # calculate and standardise the cumsum of the eval
    cumVar = eVal.cumsum() / eVal.sum()
    # find the index of last cumsum that < varThres
    dim = cumVar.values.searchsorted(varThres)
    # [0: dim] are the eVal and eVec important
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[:, : dim + 1]
    return eVal, eVec

def orthoFeats(dfX, varThres=.95):
    """
    Given a dataframe dfX of features, compute orthofeatures dfP
    :params dfX: pd.df, features
    :params varThres: float, threshold to select the significant Principal components

    :return
        dfP: pd.df, orthofeatures
    """
    # standardized features
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    # calculate the ZZ`(dot)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    # find the (significant) eVal and eVec
    eVal, eVec = get_eVec(dot, varThres)
    # get the orthofeatures
    dfP = np.dot(dfZ, eVec)
    return dfP


# COMPUTATION OF WEIGHTED KENDALL’S TAU BETWEEN FEATURE IMPORTANCE AND INVERSE PCA RANKING
# >>> import numpy as np
# >>> from scipy.stats import weightedtau
# >>> featImp=np.array([.55,.33,.07,.05]) # feature importance
# >>> pcRank=np.array([1,2,4,3]) # PCA rank
# >>> weightedtau(featImp,pcRank**-1.)[0]

def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    # trnsX = X, cont = y, informative features = `n_informative`, redundant features = `n_redundant`
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                      n_redundant=n_redundant, random_state=0, shuffle=False)
    # n_samples days, freq = Business days, end by today
    df0 = pd.to_datetime(pd.date_range(end = pd.Timestamp.today(), periods = n_samples, freq = 'B').date)
    #df0 = pd.date_range(periods = n_samples, freq = pd.tseries.offsets.BDay(), end = pd.datetime.today())
    # transform trnsX and cont into pd.df, cont.column.name = "bins"
    trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(cont, index=df0).to_frame('bin')
    # first n_informative are informative features, after that, first (n_redundant) are redundant features
    df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    # the rest are noise features
    df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
    # set trnsX.columns name
    trnsX.columns = df0
    # equal weight for each label
    cont['w'] = 1. / cont.shape[0]
    # vertical barrier is cont.index it self
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


def featImportance(trnsX, cont,
                   n_estimators=1000, cv=10, max_samples=1., numThreads=24, pctEmbargo=0,
                   scoring='accuracy', method='SFI', minWLeaf=0., **kargs):
    # feature importance from a random forest
    # bagged decision trees as default classifier
    n_jobs = (-1 if numThreads > 1 else 1)  # run 1 thread with ht_helper in dirac1
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1, class_weight='balanced',
                                 min_weight_fraction_leaf=minWLeaf)
    # A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
    # 'oob_score = True' use out-of-bag samples to estimate the generalization error.
    clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_features=1., max_samples=max_samples,
                            oob_score=True, n_jobs=n_jobs)
    # fit the data
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    # oob = Score of the training dataset obtained using an out-of-bag estimate.
    oob = fit.oob_score_
    if method == 'MDI':  # MDI
        # MDI feature importance
        imp = featImpMDI(fit, featNames = trnsX.columns)
        # oos = CV
        oos = cvScore(clf, X = trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo,
                      scoring=scoring).mean()
    elif method == 'MDA':  # MDA
        # MDA feature importance and oos
        imp, oos = featImpMDA(clf, X = trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'],
                              pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':  # SFI
        # CV
        cvGen = PurgedKFold(n_splits = cv, t1 = cont['t1'], pctEmbargo = pctEmbargo)
        # oos score
        oos = cvScore(clf, X = trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen).mean()
        clf.n_jobs = 1  # paralellize auxFeatImpSFI rather than clf
        # find the SFI imp
        imp = mpPandasObj(auxFeatImpSFI, ('featNames', trnsX.columns), numThreads, clf=clf, trnsX=trnsX, cont=cont,
                          scoring=scoring, cvGen=cvGen)
    return imp, oob, oos

def testFunc(n_features = 40, n_informative = 10, n_redundant = 10,
             n_estimators = 1000, n_samples = 10000, cv = 10):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features—n_informative—n_redundant
    # generate synthetic data
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    # arguments
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'], 'max_samples': [1.]}
    # split dict0 into 3 different jobs (by method)
    jobs = (dict(zip(dict0, i)) for i in product(*dict0.values()))
    out = []  # empty list
    # key arguments
    kargs = {'pathOut': './testFunc/', 'n_estimators': n_estimators, 'tag': 'testFunc', 'cv': cv}
    for job in jobs:  # for each jobs
        # job params
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_' + '%.2f' % job['minWLeaf'] + '_' + str(
            job['max_samples'])
        print(job['simNum'])  # print job params
        kargs.update(job)  # update/add the elemets to the dictionary
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)  # find faeture importance using imp, oob, oos
        plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)  # plot the feature importance
        df0 = imp[['mean']] / imp['mean'].abs().sum()  # normalised
        df0['type'] = [i[0] for i in df0.index]  #
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({'oob': oob, 'oos': oos})  # update/add the elemets to the dictionary
        df0.update(job)  # update/add the elemets to the dictionary
        out.append(df0)  # append df0 to out
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])  # sort the df by
    #     # only the followings are output
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    #    # out = out['method', 'scoring', 'minWLeaf', 'max_samples', 'oob', 'oos']
    out.to_csv(kargs['pathOut'] + 'stats.csv')
    return

# =================================================================================================================
#      Hyper Parameter Tuning
# =================================================================================================================

class MyPipeline(Pipeline):
    """
    Inherit all methods from sklearn's `Pipeline`
    Overwrite the inherited `fit` method with a new one that handles the argument `sample weight`
    After which it redirects to the parent class
    """
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)

def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid,
                cv = 3, bagging = [0, None, 1.], n_jobs = -1, pctEmbargo = 0, **fit_params):
    """
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params:
    :return gs:
    """
    scoring = 'f1' # f1 for meta-labeling
    # if set(lbl.values) == {0, 1}: # if label values are 0 or 1
    #     scoring = 'f1' # f1 for meta-labeling
    # else:
    #     scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    # perform grid search
    gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                        scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    # best estimator and the best parameter
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0: # max_samples > 0
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

def clfHyperFitRand(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0,None,1.], rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    """
    Randimised Search with Purged K-fold CV
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers, used for PurgedKFold
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params rndSearchIter
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params:
    :return gs:
    """
    if set(lbl.values) == {0,1}:# if label values are 0 or 1
        scoring = 'f1' # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    if rndSearchIter == 0: # randomised grid search
        gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                            scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    else: # normal grid search
        gs = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid,
                                scoring = scoring, cv = inner_cv, n_jobs = n_jobs,
                                iid = False, n_iter = rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

from scipy.stats import rv_continuous,kstest
class logUniform_gen(rv_continuous):
# random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
def logUniform(a = 1,b = np.exp(1)) : return logUniform_gen(a = a,b = b,name = 'logUniform')

# =================================================================================================================
#      Plot Chart
# =================================================================================================================

def plot_bar_counts(tick, volume, dollar):
    """
    Tick, Volume, Dollar Bard의 counting 횟수를 plotting하는 함수입니다
    """
    f, ax = plt.subplots(figsize=(15, 5))
    tick.plot(ax=ax, ls='-', label='tick count')
    volume.plot(ax=ax, ls='--', label='volume count')
    dollar.plot(ax=ax, ls='-.', label='dollar count')
    ax.set_title('Scaled Bar Counts')
    ax.legend()
    return

def plot_hist(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 6))
    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        g = sns.distplot(bar, ax=axes[i], kde=False, label=typ)
        g.set(yscale='log')
        axes[i].legend()
    plt.tight_layout()
    return

def plotSampleData(ref, sub, bar_type, *args, **kwds):
    """
    Sampling된 Data를 Plotting합니다
    """

    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend()
    ref.plot(*args, **kwds, ax=axes[1], marker='o', label='price')
    sub.plot(*args, **kwds, ax=axes[2], marker='X', ls='',
             color='r', label=bar_type)
    for ax in axes[1:]: ax.legend()
    plt.tight_layout()
    return

def plot_autocorr(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 7))

    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        sm.graphics.tsa.plot_acf(bar, lags=120, ax=axes[i],
                                 alpha=0.05, unbiased=True, fft=True,
                                 zero=False,
                                 title=f'{typ} AutoCorr')
    plt.tight_layout()
    return

def plotWeights(dRange, nPlots, size):
    """
    Weight를 Plotting하는 함수입니다
    :param dRange:
    :param nPlots:
    :param size:
    :return:
    """
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[:: -1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left')
    plt.show()
    return

def plotMinFFD():
    """
    adfuller test를 통과하는 최소의 d값을 찾습니다
    :return:
    """
    from statsmodels.tsa.stattools import adfuller
    path = './'
    instName = 'ES1_Index_Method12'
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName + '.csv', index_col=0, parse_dates=True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last()  # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres=.01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[: 4]) + [df2[4]['5%']] + [corr]  # with critical value
    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.savefig(path + instName + '_testMinFFD.png')
    return

def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    # sort imp['mean'] from low to high
    imp = imp.sort_values('mean', ascending=True)
    # plot horizontal bar
    ax = imp['mean'].plot(kind='barh', color='b', alpha=.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    if method == 'MDI':  # for MDI
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)  # disable y-axis
    for i, j in zip(ax.patches, imp.index):  # label
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    plt.title(
        'tag=' + tag + ' | simNum= ' + str(simNum) + ' | oob=' + str(round(oob, 4)) + ' | oos=' + str(round(oos, 4)))
    plt.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi=100)
    plt.clf()
    plt.close()
    return

# =================================================================================================================
#           Performance
# =================================================================================================================

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms,
                        min(numThreads, numAtoms) + 1)  # find the indices (may not int) of the partition parts
    parts = np.ceil(parts).astype(int)  # ceil the float indices into int
    return parts

def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts = [0]
    numThreads_ = min(numThreads, numAtoms)
    for _ in range(numThreads_):
        # find the appropriate size of each part by an algorithms
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        # store part into parts
        parts.append(part)
    # rounded to the nearest natural number
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        # dont forget the 0 at the begining
        parts = np.append(np.array([0]), parts)
    return parts

def mpPandasObj(func, pdObj, numThreads = 24, mpBatches = 1, linMols = True, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    :params func: function to be parallelized. Returns a DataFrame
    :params pdObj: tuple,
        + pdObj[0]: Name of argument used to pass the molecule
        + pdObj[1]: List of atoms that will be grouped into molecules
    :params numThreads: int, no. of threads that will be used in parallel (1 processor per thread)
    :params mpBatches: int, no. of parallel batches (jobs per core)
    :params linMols: bool, whether partitions will be linear or double-nested
    :params kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    # ----------------Partition the dataset-------------------------
    # parts: the indices to separate
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        # name of argument: molecule, function: func
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)  # update kargs?
        jobs.append(job)
    # -----------------multiprocessing--------------------
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    # ------------determine the datatype of the output----
    try:
        if len(out) == 0:
            return pd.DataFrame()
        elif isinstance(out[0], pd.DataFrame):
            df0 = pd.DataFrame()
        elif isinstance(out[0], pd.Series):
            df0 = pd.Series()
        else:
            return out
        # Append the output to the df0
        for i in out:
            df0 = df0.append(i)
        # sort objects by labels
        df0 = df0.sort_index()
    except:
        print(type(out))
        df0 = pd.DataFrame()
    return df0

def processJobs_(jobs):
    # Run jobs sequentially, for debugging or numThread = 1
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    # keep us informed about the percentage of jobs completed
    # msg[0]: % completed, msg[1]: time elapses
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    # msg[2]:minutes remaining
    msg.append(msg[1] * (1 / msg[0] - 1))
    # the current time
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    # convert a list `msg` into a string `msg`
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')  # pointer goes to the front?
    else:
        sys.stderr.write(msg + '\n')  # pointer goes to the next line
    return

def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)  # i7 I cores..should delete 'numThreads' really
    # 'map': map the function to the arguments/parameters
    # 'pool.map': parallelise `expandCall`
    # 'imap_unordered`: iterators, results will be yielded as soon as they are ready, regardless of the order of the input iterable
    outputs = pool.imap_unordered(expandCall, jobs)  # 'imap_unordered` seems to use less memory than 'imap'
    out = []
    time0 = time.time()
    # Process asyn output, report progress
    # I guess the results are actually output here
    for i, out_ in enumerate(outputs, 1):  # index start at 1
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()  # close the pool, stop accepting new jobs
    pool.join()  # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    # Unwrap the items(atoms) in the job(molecule) and execute the callback function
    func = kargs['func']  # function
    del kargs['func']  # delete the `function` column/argument
    out = func(**kargs)  # put the arguments into the function
    return out

def processJobsRedux(jobs, task=None, cpus=4, redux=None, reduxArgs={}, reduxInPlace=False):
    '''
    Run in parallel
    jobs must contain a ’func’ callback, for expandCall
    redux prevents wasting memory by reducing output on the fly
    :params redux: func, a callback to the function that carries out the reduction, e.g. pd.DataFrame.add
    :params reduxArgs: dict, contains the keyword arguments that must be passed to the redux (if any)
        e.g. if redux = 'od,DataFrame.join, reduxArg = {'how':'outer'}
    :params reduxInPlace: bool, indicate whether the redux operation should happen in-place or not
        e.g. redux = dict.update or redux = list.append requires reduxInplace = True
            because updating a dictionary or appending a list is both in-place operations
    '''

    if task is None:  # get the name of the function/tasl
        task = jobs[0]['func'].__name__
    # 'map': map the function to the arguments/parameters
    # 'pool.map': parallelise `expandCall`
    # 'imap_unordered`: iterators, results will be yielded as soon as they are ready, regardless of the order of the input iterable
    pool = mp.Pool(processes=cpus)
    imap = pool.imap_unordered(expandCall, jobs)
    out = None
    time0 = time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(imap, 1):
        if out is None:  # the first element
            if redux is None:  # if the reduction function is not specified
                out = [out_]
                redux = list.append
                reduxInPlace = True
            else:
                out = copy.deepcopy(out_)
        else:  # not the first
            if reduxInPlace:  # if inplace, no need to re-assign to out
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), time0, task)
    pool.close()  # close the pool, stop accepting new jobs
    pool.join()  # this is needed to prevent memory leaks
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out

def mpJobList(func, argList, numThreads, mpBatches = 1, linMols = True,
              redux = None, reduxArgs = {}, reduxInPlace = False, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    :params func: function to be parallelized. Returns a DataFrame
    :params argList: tuple,
        + argList[0]: Name of argument used to pass the molecule
        + argList[1]: List of atoms that will be grouped into molecules
    :params mpBatches: int, no. of parallel batches (jobs per core)
    :params linMols: bool, whether partitions will be linear or double-nested
    :params redux: func, a callback to the function that carries out the reduction, e.g. pd.DataFrame.add
    :params reduxArgs: dict, contains the keyword arguments that must be passed to the redux (if any)
        e.g. if redux = 'od,DataFrame.join, reduxArg = {'how':'outer'}
    :params reduxInPlace: bool, indicate whether the redux operation should happen in-place or not
        e.g. redux = dict.update or redux = list.append requires reduxInplace = True
            because updating a dictionary or appending a list is both in-place operations

    Example: df1=mpJobList(func,('molecule',df0.index),24)
    '''

    # ----------------Partition the dataset-------------------------
    # parts: the indices to separate
    if numThreads:
        cpus = numThreads
    else:
        if platform.system() == 'Windows':
            cpus = 1
        else:
            cpus = cpu_count() - 1

    if linMols:
        parts = linParts(len(argList[1]), cpus * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), cpus * mpBatches)
    jobs = []

    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i - 1]: parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    # -----------------multiprocessing--------------------
    out = processJobsRedux(jobs, redux=redux, reduxArgs=reduxArgs,
                           reduxInPlace=reduxInPlace, cpus=cpus)
    # no need to process an outputed list, save memory and time
    return out