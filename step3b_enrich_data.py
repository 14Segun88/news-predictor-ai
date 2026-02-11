#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 3b: ПОЛНОЕ ОБОГАЩЕНИЕ ДАННЫХ v3                   ║
║  Уровень 1: Из датасета (фундамент)                    ║
║  Уровень 2: Рыночные данные (S&P500, DXY, VIX, Oil)   ║
║  Уровень 3: Тех.анализ + Объёмы + Волны + Позиции     ║
╚══════════════════════════════════════════════════════════╝

Запуск: python3 step3b_enrich_data.py
Результат: data/enriched_calendar.csv
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "data/clean_calendar.csv"
OUTPUT_FILE = "data/enriched_calendar.csv"
MARKET_CACHE = "data/market_data_v3.csv"  # v3 с Volume!


# ═══════════════════════════════════════════════════
# ТЕХНИЧЕСКИЙ АНАЛИЗ: ручной расчёт индикаторов
# ═══════════════════════════════════════════════════

def calc_rsi(series, period=14):
    """RSI — Relative Strength Index (0-100). >70 = перекуплено, <30 = перепродано"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    """MACD — Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series, period=20, std_mult=2):
    """Bollinger Bands — показывает волатильность"""
    sma = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().fillna(0)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Позиция цены внутри канала (0=нижняя, 1=верхняя)
    bb_position = (series - lower) / (upper - lower + 1e-10)
    bb_width = (upper - lower) / (sma + 1e-10) * 100  # ширина в %
    return bb_position, bb_width


def calc_trend_direction(series, short=5, long=20):
    """Направление тренда: +1=бычий, -1=медвежий, 0=боковой"""
    sma_short = series.rolling(short, min_periods=1).mean()
    sma_long = series.rolling(long, min_periods=1).mean()
    trend = np.where(sma_short > sma_long * 1.005, 1,
                     np.where(sma_short < sma_long * 0.995, -1, 0))
    return trend


def calc_swing_position(series, window=20):
    """Позиция цены относительно последнего swing high/low"""
    rolling_high = series.rolling(window, min_periods=1).max()
    rolling_low = series.rolling(window, min_periods=1).min()
    # Где мы между экстремумами (0 = у low, 1 = у high)
    position = (series - rolling_low) / (rolling_high - rolling_low + 1e-10)
    return position


def calc_wave_count(series, threshold_pct=1.0):
    """Упрощённый подсчёт волн (кол-во разворотов за 20 дней)"""
    pct_change = series.pct_change() * 100
    # Считаем значимые развороты (> threshold%)
    direction = np.sign(pct_change)
    reversals = (direction.diff().abs() > 1).astype(int)
    wave_count = reversals.rolling(20, min_periods=1).sum()
    return wave_count


def download_market_data_v3(start_date, end_date):
    """Скачиваем рыночные данные С ОБЪЁМАМИ"""
    
    if os.path.exists(MARKET_CACHE):
        print("   📁 Рыночные данные v3 найдены в кэше")
        return pd.read_csv(MARKET_CACHE, parse_dates=['Date'])
    
    print("   ⬇️  Скачиваю рыночные данные с объёмами (yfinance)...")
    import yfinance as yf
    
    tickers = {
        '^GSPC': 'SP500',
        'DX-Y.NYB': 'DXY',
        '^VIX': 'VIX',
        'CL=F': 'Oil_WTI',
        '^TNX': 'US10Y',
    }
    
    frames = []
    
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) == 0:
                print(f"      ⚠️  {name}: нет данных")
                continue
            
            # Flatten MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            chunk = pd.DataFrame(index=data.index)
            chunk[f'{name}_Close'] = data['Close'].values
            chunk[f'{name}_Change'] = data['Close'].pct_change().values * 100
            
            has_vol = 'Volume' in data.columns and data['Volume'].sum() > 0
            if has_vol:
                chunk[f'{name}_Volume'] = data['Volume'].values.astype(float)
            
            frames.append(chunk)
            print(f"      ✅ {name}: {len(data)} дней" + 
                  (f" (Volume: ✅)" if has_vol else " (Volume: ❌)"))
        except Exception as e:
            print(f"      ❌ {name}: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    # Объединяем по индексу (дате)
    market = pd.concat(frames, axis=1)
    market = market.reset_index()
    market = market.rename(columns={market.columns[0]: 'Date'})
    market['Date'] = pd.to_datetime(market['Date'])
    
    market.to_csv(MARKET_CACHE, index=False)
    print(f"   💾 Кэш v3 сохранён: {MARKET_CACHE}")
    
    return market


def add_ta_features(market):
    """Добавляем технический, объёмный и волновой анализ"""
    
    instruments = ['SP500', 'DXY', 'VIX', 'Oil_WTI', 'US10Y']
    
    for name in instruments:
        close_col = f'{name}_Close'
        if close_col not in market.columns:
            continue
        
        close = market[close_col].astype(float)
        
        # ═══ ТЕХНИЧЕСКИЙ АНАЛИЗ ═══
        # RSI(14)
        market[f'{name}_RSI'] = calc_rsi(close, 14)
        
        # MACD
        macd_line, signal, histogram = calc_macd(close)
        market[f'{name}_MACD_Hist'] = histogram
        
        # Bollinger Bands
        bb_pos, bb_width = calc_bollinger(close, 20, 2)
        market[f'{name}_BB_Position'] = bb_pos   # 0-1
        market[f'{name}_BB_Width'] = bb_width     # волатильность
        
        # SMA
        market[f'{name}_SMA20'] = close.rolling(20, min_periods=1).mean()
        market[f'{name}_SMA50'] = close.rolling(50, min_periods=1).mean()
        market[f'{name}_Above_SMA20'] = (close > market[f'{name}_SMA20']).astype(int)
        market[f'{name}_Above_SMA50'] = (close > market[f'{name}_SMA50']).astype(int)
        
        # ═══ ВОЛНОВОЙ АНАЛИЗ ═══
        # Направление тренда
        market[f'{name}_Trend'] = calc_trend_direction(close, 5, 20)
        
        # Swing Position (где мы между экстремумами)
        market[f'{name}_Swing_Pos'] = calc_swing_position(close, 20)
        
        # Wave Count (кол-во разворотов)
        market[f'{name}_Wave_Count'] = calc_wave_count(close, 1.0)
        
        # Consecutive Days (дней подряд в одном направлении)
        change = close.diff()
        direction = np.sign(change)
        groups = (direction != direction.shift()).cumsum()
        market[f'{name}_Consec_Days'] = close.groupby(groups).cumcount() + 1
        market[f'{name}_Consec_Dir'] = direction
        
        # ═══ ОБЪЁМНЫЙ АНАЛИЗ ═══
        vol_col = f'{name}_Volume'
        if vol_col in market.columns:
            volume = market[vol_col].astype(float)
            
            # Volume MA (20)
            vol_ma = volume.rolling(20, min_periods=1).mean()
            market[f'{name}_Vol_Ratio'] = volume / (vol_ma + 1)  # >1 = выше среднего
            
            # Volume Spike (>2x среднего)
            market[f'{name}_Vol_Spike'] = (volume > vol_ma * 2).astype(int)
            
            # Volume Trend (растёт/падает)
            market[f'{name}_Vol_Trend'] = volume.rolling(5, min_periods=1).mean() / (vol_ma + 1)
            
            # ═══ ПОЗИЦИОНИРОВАНИЕ (PROXY) ═══
            # On-Balance Volume (OBV) — кумулятивный объём направления
            obv_direction = np.where(close.diff() > 0, volume, 
                           np.where(close.diff() < 0, -volume, 0))
            obv = pd.Series(obv_direction).cumsum()
            obv_ma = obv.rolling(20, min_periods=1).mean()
            market[f'{name}_OBV_Signal'] = np.sign(obv - obv_ma)  # +1=покупка, -1=продажа
            
            # Money Flow (Price × Volume direction)
            mf = close * volume
            mf_pos = mf.where(close.diff() > 0, 0).rolling(14, min_periods=1).sum()
            mf_neg = mf.where(close.diff() < 0, 0).rolling(14, min_periods=1).sum()
            market[f'{name}_MFI'] = 100 - 100 / (1 + mf_pos / (mf_neg + 1))  # Money Flow Index
            
            # Volume-Weighted Price Momentum (прокси институционального позиционирования)
            vwap_5 = (close * volume).rolling(5, min_periods=1).sum() / (volume.rolling(5, min_periods=1).sum() + 1)
            vwap_20 = (close * volume).rolling(20, min_periods=1).sum() / (volume.rolling(20, min_periods=1).sum() + 1)
            market[f'{name}_VWAP_Signal'] = np.sign(vwap_5 - vwap_20)  # институционалы покупают или продают
        else:
            # Нет объёмов — заполняем нулями
            for suffix in ['Vol_Ratio', 'Vol_Spike', 'Vol_Trend', 'OBV_Signal', 'MFI', 'VWAP_Signal']:
                market[f'{name}_{suffix}'] = 0
    
    return market


def enrich_data():
    print("╔══════════════════════════════════════════════════╗")
    print("║  🔬 ПОЛНОЕ ОБОГАЩЕНИЕ ДАННЫХ v3                 ║")
    print("║  Level 1: Dataset  |  Level 2: Market Data      ║")
    print("║  Level 3: TA + Volume + Wave + Positioning      ║")
    print("╚══════════════════════════════════════════════════╝")
    
    df = pd.read_csv(INPUT_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    original_cols = len(df.columns)
    print(f"\n📦 Исходные данные: {len(df)} строк, {original_cols} колонок")
    
    # ═══════════════════════════════════════════════════
    # УРОВЕНЬ 1: ИЗ ДАТАСЕТА
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("📊 УРОВЕНЬ 1: Фичи из датасета")
    print(f"{'='*50}")
    
    df['Surprise'] = df['Actual'] - df['Forecast']
    df['Date'] = df['DateTime'].dt.date
    
    # 1.1: Surprise History
    print("\n  1️⃣  Surprise History...")
    df['Surprise_Mean_3'] = np.nan
    df['Surprise_Mean_6'] = np.nan
    df['Surprise_Std_6'] = np.nan
    df['Event_Streak'] = 0
    
    for ev in df['Event'].unique():
        m = df['Event'] == ev
        edf = df[m]
        if len(edf) < 3:
            continue
        s = edf['Surprise']
        df.loc[m, 'Surprise_Mean_3'] = s.shift(1).rolling(3, min_periods=1).mean()
        df.loc[m, 'Surprise_Mean_6'] = s.shift(1).rolling(6, min_periods=1).mean()
        df.loc[m, 'Surprise_Std_6'] = s.shift(1).rolling(6, min_periods=2).std()
        
        streak = 0
        streaks = []
        for val in s.shift(1).values:
            if pd.isna(val):
                streaks.append(0)
                continue
            streak = (max(streak, 0) + 1) if val > 0 else ((min(streak, 0) - 1) if val < 0 else 0)
            streaks.append(streak)
        df.loc[m, 'Event_Streak'] = streaks
    
    for c in ['Surprise_Mean_3', 'Surprise_Mean_6', 'Surprise_Std_6']:
        df[c] = df[c].fillna(0)
    print("     ✅ Mean3, Mean6, Std6, Streak")
    
    # 1.2: Forecast Accuracy
    print("  2️⃣  Forecast Accuracy...")
    df['Forecast_Accuracy_3'] = np.nan
    df['Forecast_Miss_Rate'] = np.nan
    
    for ev in df['Event'].unique():
        m = df['Event'] == ev
        edf = df[m]
        if len(edf) < 3:
            continue
        ae = (edf['Actual'] - edf['Forecast']).abs()
        df.loc[m, 'Forecast_Accuracy_3'] = ae.shift(1).rolling(3, min_periods=1).mean()
        scale = edf['Actual'].abs().replace(0, 1)
        miss = (ae / scale > 0.1).astype(float)
        df.loc[m, 'Forecast_Miss_Rate'] = miss.shift(1).rolling(6, min_periods=1).mean()
    
    df['Forecast_Accuracy_3'] = df['Forecast_Accuracy_3'].fillna(0)
    df['Forecast_Miss_Rate'] = df['Forecast_Miss_Rate'].fillna(0.5)
    print("     ✅ Accuracy_3, Miss_Rate")
    
    # 1.3: Trend
    print("  3️⃣  Trend + Momentum...")
    df['Actual_Trend_3'] = np.nan
    df['Actual_Momentum'] = np.nan
    
    for ev in df['Event'].unique():
        m = df['Event'] == ev
        edf = df[m]
        if len(edf) < 3:
            continue
        ad = edf['Actual'].diff()
        t3 = ad.shift(1).rolling(3, min_periods=1).mean()
        t6 = ad.shift(1).rolling(6, min_periods=1).mean()
        df.loc[m, 'Actual_Trend_3'] = t3
        df.loc[m, 'Actual_Momentum'] = t3 - t6
    
    df['Actual_Trend_3'] = df['Actual_Trend_3'].fillna(0)
    df['Actual_Momentum'] = df['Actual_Momentum'].fillna(0)
    print("     ✅ Trend_3, Momentum")
    
    # 1.4: Revision Rate
    print("  4️⃣  Revision Rate...")
    df['Revision_Rate'] = np.nan
    for ev in df['Event'].unique():
        m = df['Event'] == ev
        edf = df[m]
        if len(edf) < 3:
            continue
        prev_actual = edf['Actual'].shift(1)
        revised = (edf['Previous'] != prev_actual).astype(float)
        df.loc[m, 'Revision_Rate'] = revised.shift(1).rolling(6, min_periods=1).mean()
    df['Revision_Rate'] = df['Revision_Rate'].fillna(0)
    print("     ✅ Revision_Rate")
    
    # 1.5: Context
    print("  5️⃣  Events Context...")
    df['Week'] = df['DateTime'].dt.isocalendar().week.astype(int)
    df['Year_Week'] = df['DateTime'].dt.year.astype(str) + '_' + df['Week'].astype(str)
    wc = df.groupby('Year_Week').size().reset_index(name='Same_Week_Events')
    df = df.merge(wc, on='Year_Week', how='left')
    hi = df[df['Importance'] >= 3].groupby('Year_Week').size().reset_index(name='High_Impact_Same_Week')
    df = df.merge(hi, on='Year_Week', how='left')
    df['High_Impact_Same_Week'] = df['High_Impact_Same_Week'].fillna(0)
    ef = df['Event'].value_counts().to_dict()
    df['Event_Frequency'] = df['Event'].map(ef)
    print("     ✅ Same_Week, High_Impact, Event_Freq")
    
    # 1.6: Cross-Currency
    print("  6️⃣  Cross-Currency EUR↔USD...")
    for cur in ['USD', 'EUR']:
        other = 'EUR' if cur == 'USD' else 'USD'
        od = df[df['Currency'] == other].groupby('Date').agg({
            'Surprise': ['mean', 'count', 'std'], 'Importance': 'mean'
        }).reset_index()
        od.columns = ['Date', f'{other}_Surp_Mean', f'{other}_Events', f'{other}_Surp_Std', f'{other}_Imp']
        od[f'{other}_Surp_Std'] = od[f'{other}_Surp_Std'].fillna(0)
        m = df['Currency'] == cur
        dm = df.loc[m, ['Date']].merge(od, on='Date', how='left')
        for col in [f'{other}_Surp_Mean', f'{other}_Events', f'{other}_Surp_Std', f'{other}_Imp']:
            df.loc[m, col] = dm[col].values
    cc = [c for c in df.columns if c.startswith(('EUR_', 'USD_'))]
    df[cc] = df[cc].fillna(0)
    print(f"     ✅ {len(cc)} cross-currency фичей")
    
    # 1.7: Lags
    print("  7️⃣  Lag Features...")
    df['Prev_Actual_1'] = np.nan
    df['Prev_Surprise_1'] = np.nan
    df['Days_Since_Last'] = np.nan
    for ev in df['Event'].unique():
        m = df['Event'] == ev
        edf = df[m]
        if len(edf) < 2:
            continue
        df.loc[m, 'Prev_Actual_1'] = edf['Actual'].shift(1)
        df.loc[m, 'Prev_Surprise_1'] = edf['Surprise'].shift(1)
        df.loc[m, 'Days_Since_Last'] = edf['DateTime'].diff().dt.total_seconds() / 86400
    df['Prev_Actual_1'] = df['Prev_Actual_1'].fillna(df['Previous'])
    df['Prev_Surprise_1'] = df['Prev_Surprise_1'].fillna(0)
    df['Days_Since_Last'] = df['Days_Since_Last'].fillna(30)
    print("     ✅ Prev_Actual, Prev_Surprise, Days_Since")
    
    # 1.8: Seasonality
    print("  8️⃣  Seasonality...")
    df['Quarter'] = df['DateTime'].dt.quarter
    df['Is_Friday'] = (df['DateTime'].dt.dayofweek == 4).astype(int)
    df['Is_First_Week'] = (df['DateTime'].dt.day <= 7).astype(int)
    
    lvl1 = len(df.columns) - original_cols
    print(f"\n   📊 Уровень 1: +{lvl1} фичей")
    
    # ═══════════════════════════════════════════════════
    # УРОВЕНЬ 2 + 3: РЫНОЧНЫЕ ДАННЫЕ + ТА + ОБЪЁМЫ + ВОЛНЫ
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("📈 УРОВЕНЬ 2+3: Рынок + Тех.анализ + Объёмы + Волны")
    print(f"{'='*50}")
    
    min_date = df['DateTime'].min().strftime('%Y-%m-%d')
    max_date = df['DateTime'].max().strftime('%Y-%m-%d')
    
    market = download_market_data_v3(min_date, max_date)
    
    if len(market) > 0:
        market['Date'] = pd.to_datetime(market['Date'])
        
        # Добавляем TA/Volume/Wave индикаторы
        print("\n   🔧 Рассчитываю технические индикаторы...")
        market = add_ta_features(market)
        
        ta_cols = [c for c in market.columns if any(x in c for x in 
                   ['RSI', 'MACD', 'BB_', 'SMA', 'Above_', 'Trend', 'Swing', 'Wave',
                    'Consec', 'Vol_', 'OBV', 'MFI', 'VWAP'])]
        print(f"   ✅ Добавлено {len(ta_cols)} тех. индикаторов")
        
        # Сдвигаем ВСЁ на 1 день назад (чтобы не подглядывать)
        shift_cols = [c for c in market.columns if c != 'Date']
        for col in shift_cols:
            market[col] = market[col].shift(1)
        
        # Merge по дате
        market['MergeDate'] = market['Date'].dt.date
        df['MergeDate'] = pd.to_datetime(df['Date']).dt.date if not isinstance(df['Date'].iloc[0], str) else df['Date']
        
        before = len(df.columns)
        market_to_merge = market.drop(columns=['Date']).copy()
        df = df.merge(market_to_merge, on='MergeDate', how='left')
        df = df.drop(columns=['MergeDate'], errors='ignore')
        
        # Fill NaN (выходные)
        fill_cols = [c for c in df.columns if any(m in c for m in 
                     ['SP500', 'DXY', 'VIX', 'Oil', 'US10Y'])]
        df[fill_cols] = df[fill_cols].ffill().bfill().fillna(0)
        
        after = len(df.columns)
        print(f"\n   📈 Уровень 2+3: +{after - before} фичей")
    
    # ═══════════════════════════════════════════════════
    # ФИНАЛ
    # ═══════════════════════════════════════════════════
    drop_cols = ['Surprise', 'Date', 'Week', 'Year_Week']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    final = len(df.columns)
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ ОБОГАЩЕНИЕ v3 ЗАВЕРШЕНО!                             ║
╚══════════════════════════════════════════════════════════╝

  📁 Файл: {OUTPUT_FILE}
  📊 Строк: {len(df)}
  📋 Колонок: {final} (было {original_cols})
  
  🆕 Уровень 1 (датасет):     ~30 фичей
  🆕 Уровень 2 (рынок):       ~10 фичей (Close + Change)
  🆕 Уровень 3 (тех.анализ):
     📊 RSI(14) для S&P500/DXY/VIX/Oil/US10Y
     📈 MACD Histogram
     📉 Bollinger Bands (Position + Width)
     📐 SMA20/50 + Above signals
     🌊 Trend Direction + Swing Position + Wave Count
     🔊 Volume Ratio + Spike + Trend
     💰 OBV Signal + Money Flow Index + VWAP Signal
  
  Следующий шаг: python3 step4_train_model.py
""")


if __name__ == "__main__":
    enrich_data()
