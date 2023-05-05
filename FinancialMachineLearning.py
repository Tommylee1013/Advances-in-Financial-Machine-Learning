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
import statsmodels.api as sm
import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report
import FinancialMultiProcessing as fmp

from numba import jit

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm, tqdm_notebook

import warnings

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


# @jit(nopython=True)

def mad_outlier(y, thresh = 3.):
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


def bar_sampling(df, column, threshold, tick=False):
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


def plot_sample_data(ref, sub, bar_type, *args, **kwds):
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


def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))


def get_test_stats(bar_types, bar_returns, test_func, *args, **kwds):
    dct = {bar: (int(bar_ret.shape[0]), test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0: 'sample size', 1: f'{test_func.__name__}_stat'}).T)
    return df


def plot_autocorr(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 7))

    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        sm.graphics.tsa.plot_acf(bar, lags=120, ax=axes[i],
                                 alpha=0.05, unbiased=True, fft=True,
                                 zero=False,
                                 title=f'{typ} AutoCorr')
    plt.tight_layout()


def plot_hist(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types), figsize=(10, 6))
    for i, (bar, typ) in enumerate(zip(bar_returns, bar_types)):
        g = sns.distplot(bar, ax=axes[i], kde=False, label=typ)
        g.set(yscale='log')
        axes[i].legend()
    plt.tight_layout()


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

def bband(data: pd.Series, window: int = 21, width: float = 0.005):
    """
    Bollinger Band를 구축합니다
    """
    avg = data.ewm(span=window).mean()
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


def cumsum_events(df: pd.Series, limit: float):
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


def getTEvents(gRaw, h):
    """
    대칭 CUSUM filter를 적용하는 함수입니다

    Argument
    ----------------------------
    gRaw : price 계열의 pandas.Series 객체을 input으로 받습니다
    h : Volatility를 기반으로한 Horizonal Barrier를 설정하는 Argument 입니다
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < - h:
            sNeg = 0;
            tEvents.append(i)
        elif sPos > h:
            sPos = 0;
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


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


def TripleBarrier(close, events, ptSl, molecule):
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
    trgt : 수익률의 절대값으로 표현한 목표 pandas.Series 객체의 데이터를 input으로 합니다
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

    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = fmp.mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                          numThreads=numThreads, close=close, events=events, ptSl=np.array(ptSl_))
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan

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


def dropLabels(events, minPct = 0.05):
    # 예제가 부족할 경우 가중치를 적용해 레이블을 제거한다.
    for i in range(100):
        df0 = events['bin'].value_counts(normalize=True)

        if df0.min() > minPct or df0.shape[0] < 3: break

        print('dropped label', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]

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
    numCoEvents = fmp.mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                                  t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = fmp.mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'],
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
    numCoEvents = fmp.mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index, t1=out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = fmp.mpPandasObj(mpSampleWeights, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents=numCoEvents)
    return out


def getIndMatrix(barIx, t1):
    """
    지표 행렬을 구축하는 함수입니다
    :param barIx: Bar의 index를 input으로 합니다
    :param t1: pandas.Series 형태의 Vertical Barrier의 timestamps를 input으로 합니다 (index: eventStart, value: eventEnd)
    :return indM: binary matrix, 각 관측치의 Label에 price Bar가 미치는 영향을 보여줍니다
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
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
        out = fmp.processJobs_(jobs)
    else:
        out = fmp.processJobs(jobs, numThreads = numThreads)
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
    numCoEvents = fmp.mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx=close.index,
                              t1=events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['w'] = fmp.mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents=numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()  # normalised, sum up to sample size

    return out


def getConcurUniqueness(close, events, numThreads):
    out = events[['t1']].copy(deep = True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = fmp.mpPandasObj(getConcurrentBar, ('molecule', events.index), numThreads, closeIdx = close.index, t1 = out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = fmp.mpPandasObj(getAvgLabelUniq, ('molecule', events.index), numThreads, t1=out['t1'], numCoEvents = numCoEvents)
    out['w'] = fmp.mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1=events['t1'], numCoEvents = numCoEvents,
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
    out = fmp.mpJobList(mpGetOptimalFFD, ('molecules', d), redux=pd.DataFrame.append, data=data)

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