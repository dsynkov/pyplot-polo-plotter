import os
import time
import numpy as np
from datetime import datetime 
from poloniex import Poloniex

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mpl_finance as mpl

POLO_KEY = os.environ.get('POLO_KEY')
POLO_SECRET = os.environ.get('POLO_SECRET')

conn = Poloniex(POLO_KEY, POLO_SECRET)


def to_timestamp(string, fmt="%Y-%m-%d %H:%M:%S"):
    dt = datetime.strptime(string, fmt)
    t_tuple = dt.timetuple()
    return int(time.mktime(t_tuple))


def get_historical_data(pair, period, start, end):
    
    historical_data = conn.returnChartData(pair, period, start, end)
    
    return historical_data


def ohlc_to_dict(chart_data):
    
    # Instantiate empty lists
    close_list = []
    open_list = []
    weighted_average_list = []
    low_list = []
    quote_volume_list = []
    volume_list = []
    date_list = []
    high_list = []
    
    for item in chart_data:
        close_list.append(float(item['close']))
        open_list.append(float(item['open']))
        weighted_average_list.append(float(item['weightedAverage']))
        low_list.append(float(item['low']))
        quote_volume_list.append(float(item['quoteVolume']))
        volume_list.append(float(item['volume']))
        date_list.append(datetime.fromtimestamp(item['date']))
        high_list.append(float(item['high']))

    ohlc_dict = {
        'close': np.asarray(close_list),
        'open': np.asarray(open_list),
        'weightedAverage': np.asarray(weighted_average_list),
        'low': np.asarray(low_list),
        'quoteVolume': np.asarray(quote_volume_list),
        'volume': np.asarray(volume_list),
        'date': np.asarray(date_list),
        'high': np.asarray(high_list)
    }

    ohlc_dict['date'] = mdates.date2num(ohlc_dict['date'])
    
    return ohlc_dict


def get_coin_data(pair, period, start, end):
    
    # Convert start, end strings to Unix timestamps
    start_date, end_date = to_timestamp(start), to_timestamp(end)
    
    # Retrieve chart data from Poloniex API
    chart_data = get_historical_data(pair, period, start_date, end_date)
    
    return ohlc_to_dict(chart_data)


def moving_average(values, window):
    
    weights = np.repeat(1, window)/window
    
    return np.convolve(values, weights, 'valid')


def exp_moving_average(values, window=9):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum() 
    ema = np.convolve(values, weights, mode='full')[:len(values)]
    ema[:window] = ema[window]
    return ema


def get_macd(x, slow=26, fast=12):

    emaslow = exp_moving_average(x, slow)
    emafast = exp_moving_average(x, fast)
    
    return emaslow, emafast, emafast-emaslow, slow, fast


def get_rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi, n


def get_candle_array(coin):
    
    x = 0
    y = len(coin['date'])

    candle_array = []
    while x < y:
        append_line = coin['date'][x], coin['open'][x], coin['close'][x], coin['high'][x], coin['low'][x], coin['volume'][x]
        candle_array.append(append_line)
        x += 1
        
    return candle_array


def local_max_min(coin):
    
    y_max = max(coin['close'])
    x_max_pos = list(coin['close']).index(y_max)
    x_max = coin['date'][x_max_pos]
    
    y_min = min(coin['close'])
    x_min_pos = list(coin['close']).index(y_min)
    x_min = coin['date'][x_min_pos]   
    
    y_max, y_min = f'{y_max:.8f}', f'{y_min:.8f}'
    
    return x_max, y_max, x_min, y_min


def last_price(coin):
    
    y_last = coin['close'][-1]
    x_last_pos = list(coin['close']).index(y_last) - 1
    x_last = coin['date'][x_last_pos]
    
    y_last = f'{y_last:.8f}'
    
    return x_last, y_last


def volume_pos_neg(coin):
    
    pos = coin['open'] - coin['close'] < 0
    neg = coin['open'] - coin['close'] > 0
    
    return pos, neg


def macd_pos_neg(macd, ema9):
    
    pos = macd - ema9 < 0
    neg = macd - ema9 > 0
    
    return pos, neg


def plot_coin(pair, period, start, end, export=False):
    
    # Retrieve coin data from Polo API
    coin = get_coin_data(pair, period, start, end)
    
    # Set the plot style 
    plt.style.use('classic')
    
    # Set some other vars 
    bar_width = period / 86400
    candle_width = bar_width - .005
    bg_color = '#07000D'
    
    # Begin plot configurations
    fig = plt.figure(facecolor=bg_color)
    
    """PRICE SUBPLOT"""

    ax1 = plt.subplot2grid((7, 4), (1, 0), rowspan=4, colspan=4, facecolor=bg_color)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune='both'))
    ax1.yaxis.label.set_color('w')
    ax1.xaxis.label.set_color('w')
    ax1.spines['bottom'].set_color('w')
    ax1.spines['top'].set_color('w')
    ax1.spines['left'].set_color('w')
    ax1.spines['right'].set_color('w')
    ax1.tick_params(axis='y', colors='w')
    ax1.tick_params(axis='x', colors='w')
    ax1.grid(True, color='w')
    
    # Annotate lowest and highest
    x_max, y_max, x_min, y_min = local_max_min(coin)
    x_last, y_last = last_price(coin)
    
    ax1.annotate(y_max, xy=(x_max, y_max), va='top', ha='left', clip_on=True, color='w')
    ax1.annotate(y_min, xy=(x_min, y_min), va='bottom', ha='left', clip_on=True, color='w')
    ax1.annotate(y_last, xy=(x_last, y_last), va='top', ha='right', clip_on=True, color='w')
    
    """VOLUME SUBPLOT"""

    pos, neg = volume_pos_neg(coin)
    ax2 = plt.subplot2grid((7, 4), (5, 0), sharex=ax1, rowspan=1, colspan=4, facecolor=bg_color)
    ax2.bar(coin['date'][pos], coin['volume'][pos], color='g', width=bar_width, align='center')
    ax2.bar(coin['date'][neg], coin['volume'][neg], color='r', width=bar_width, align='center')
    ax2.yaxis.label.set_color('w')
    ax2.spines['bottom'].set_color('w') 
    ax2.spines['top'].set_color('w') 
    ax2.spines['left'].set_color('w') 
    ax2.spines['right'].set_color('w') 
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.axes.xaxis.set_ticklabels([])
    ax2.yaxis.label.set_color('w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
    ax2.text(.005, .95, 'VOLUME', va='top', transform=ax2.transAxes, color='w')
    ax2.grid(False)
    
    """MACD"""
    
    ax3 = plt.subplot2grid((7, 4), (6, 0), sharex=ax1, rowspan=1, colspan=4, facecolor=bg_color)
    
    # Get MACD data
    emaslow, emafast, macd, slow, fast = get_macd(coin['close'])
    ema9 = exp_moving_average(macd)  # Default window is 9
    macd_pos, macd_neg = macd_pos_neg(macd, ema9)
    
    ax3.plot(coin['date'], macd, color='yellow', lw=2)
    ax3.plot(coin['date'], ema9, color='#EE82EE', lw=1)
    ax3.bar(coin['date'][macd_pos], (macd-ema9)[macd_pos], color='red', width=bar_width, align='center')
    ax3.bar(coin['date'][macd_neg], (macd-ema9)[macd_neg], color='green', width=bar_width, align='center')
    ax3.spines['bottom'].set_color('w') 
    ax3.spines['top'].set_color('w') 
    ax3.spines['left'].set_color('w') 
    ax3.spines['right'].set_color('w') 
    ax3.tick_params(axis='x', colors='w')
    ax3.tick_params(axis='y', colors='w')
    ax3.yaxis.label.set_color('w')
    ax3.xaxis.label.set_color('w')
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
    ax3.text(.005, .95, 'MACD %s %s' % (fast, slow),
             va='top', transform=ax3.transAxes, color='w')
    
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10, prune='upper'))  # Format dates; max of 10 days show up
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # Date format
        
    """RSI SUBPLOT"""
    
    # Set RSI colors
    rsi_color = 'w'
    rsi_pos_color = '#386D13'
    rsi_neg_color = '#8F2020'
    
    # Retrieve RSI data
    rsi, rsi_n = get_rsi(coin['close'])
    
    # Configure plot
    ax0 = plt.subplot2grid((7, 4), (0, 0), sharex=ax1, rowspan=1, colspan=4, facecolor=bg_color)
    ax0.plot(coin['date'], rsi, rsi_color, linewidth=1.5)
    ax0.axhline(70, color=rsi_neg_color)  # Upper bound RSI
    ax0.axhline(30, color=rsi_pos_color)  # Lower bound RSI
    ax0.fill_between(coin['date'], rsi, 70, where=(rsi >= 70), facecolor=rsi_neg_color, edgecolor=rsi_neg_color)
    ax0.fill_between(coin['date'], rsi, 30, where=(rsi <= 30), facecolor=rsi_pos_color, edgecolor=rsi_pos_color)
    ax0.spines['bottom'].set_color('w') 
    ax0.spines['top'].set_color('w') 
    ax0.spines['left'].set_color('w') 
    ax0.spines['right'].set_color('w') 
    ax0.tick_params(axis='x', colors='w')
    ax0.tick_params(axis='y', colors='w')
    ax0.set_yticks([30, 70])  # RSI boundaries
    ax0.yaxis.label.set_color('w')
    ax0.set_ylim(0, 100)
    ax0.text(.005, .95, 'RSI %s' % rsi_n, va='top',
             transform=ax0.transAxes, color='w')
    
    """CANDLESTICK"""
    
    mpl.candlestick_ochl(ax1, get_candle_array(coin), width=candle_width, colorup='g', colordown='r')
    
    """FORMATTING"""  
    
    # Configure spacing
    plt.subplots_adjust(left=.09, bottom=.18, right=.94, top=.95, wspace=.20, hspace=0)
        
    # Add labels and title
    title = "%s from %s to %s" % (pair, start, end)
    plt.suptitle(title, color='w')
    
    # Make unwanted axes invisible
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=True)
    
    # Set all axes to right
    ax0.yaxis.set_ticks_position('right')
    ax1.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax3.yaxis.set_ticks_position('right')
    
    # Set the x-axis limits; fig size
    plt.xlim(coin['date'][0], max(coin['date']))
    fig.set_size_inches(20, 7.5)

    # Save figure if desired
    if export:
        fig.savefig(title + '.png', facecolor=fig.get_facecolor())

    plt.show()
