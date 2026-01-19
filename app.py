import os

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
import streamlit as st
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


INSTRUMENT_SETTINGS = {
    "EURUSD=X": {
        "default_profile": "Консервативный",
        "profiles": {
            "Консервативный": {
                "horizon": 7,
                "lower_q": 0.33,
                "upper_q": 0.66,
                "cls_conf_default": 0.55,
            },
            "Агрессивный": {
                "horizon": 5,
                "lower_q": 0.40,
                "upper_q": 0.60,
                "cls_conf_default": 0.50,
            },
        },
    },
    "GBPUSD=X": {
        "default_profile": "Консервативный",
        "profiles": {
            "Консервативный": {
                "horizon": 5,
                "lower_q": 0.30,
                "upper_q": 0.70,
                "cls_conf_default": 0.60,
            },
            "Агрессивный": {
                "horizon": 3,
                "lower_q": 0.35,
                "upper_q": 0.65,
                "cls_conf_default": 0.55,
            },
        },
    },
    "USDJPY=X": {
        "default_profile": "Консервативный",
        "profiles": {
            "Консервативный": {
                "horizon": 7,
                "lower_q": 0.33,
                "upper_q": 0.66,
                "cls_conf_default": 0.55,
            },
            "Агрессивный": {
                "horizon": 5,
                "lower_q": 0.38,
                "upper_q": 0.62,
                "cls_conf_default": 0.55,
            },
        },
    },
}


def load_price_data(ticker: str, years: int = 5, interval: str = "1d") -> pd.DataFrame | None:
    try:
        if interval in ["1h", "4h"]:
            days = max(30, min(365 * years, 730))
            df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close_df = df["Close"]
        except KeyError:
            close_df = df.iloc[:, 0]
        if isinstance(close_df, pd.DataFrame):
            close_series = close_df.iloc[:, 0]
        else:
            close_series = close_df
        df = pd.DataFrame({"Close": close_series})
    else:
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close_series = df["Close"]
    df["Returns"] = close_series.pct_change()
    df["DayOfWeek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_month"] = df.index.day
    df["week"] = df.index.isocalendar().week
    if "High" in df.columns:
        high_base = df["High"]
    else:
        high_base = close_series
    if "Low" in df.columns:
        low_base = df["Low"]
    else:
        low_base = close_series
    df["High_20"] = high_base.rolling(window=20).max()
    df["Low_20"] = low_base.rolling(window=20).min()
    for span in [8, 13, 21, 34, 55, 100]:
        df[f"EMA_{span}"] = close_series.ewm(span=span, adjust=False).mean()
    window_rsi = 14
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(window_rsi).mean()
    roll_down = loss.rolling(window_rsi).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    fast = close_series.ewm(span=12, adjust=False).mean()
    slow = close_series.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_diff"] = macd - signal
    prev_close = close_series.shift(1)
    tr1 = high_base - low_base
    tr2 = (high_base - prev_close).abs()
    tr3 = (low_base - prev_close).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = df["TR"].rolling(window=14).mean()
    return df


def get_atr_volatility_info(df: pd.DataFrame):
    if "ATR_14" not in df.columns:
        return None
    atr_series = df["ATR_14"].dropna()
    if atr_series.empty:
        return None
    current = float(atr_series.iloc[-1])
    q_low = float(atr_series.quantile(0.33))
    q_high = float(atr_series.quantile(0.66))
    if current <= q_low:
        level = "низкая"
    elif current >= q_high:
        level = "высокая"
    else:
        level = "средняя"
    return {"current": current, "q_low": q_low, "q_high": q_high, "level": level}


def add_targets(
    df: pd.DataFrame,
    horizon: int = 7,
    lower_q: float = 0.33,
    upper_q: float = 0.66,
) -> pd.DataFrame:
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    df["Future_Price_7d"] = close_series.shift(-horizon)
    df["Future_Return_7d"] = (df["Future_Price_7d"] - close_series) / close_series
    returns = df["Future_Return_7d"].dropna()
    if not returns.empty:
        lower, upper = returns.quantile([lower_q, upper_q]).values
    else:
        lower = upper = 0.0
    ret = df["Future_Return_7d"]
    df["Target_Class"] = 0
    df.loc[(ret <= lower) & (ret < 0), "Target_Class"] = -1
    df.loc[(ret >= upper) & (ret > 0), "Target_Class"] = 1
    df["Target_Reg"] = df["Future_Price_7d"]
    df = df.dropna()
    return df


def build_lstm_sequences(df: pd.DataFrame, window: int = 20):
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    values = close_series.values.astype("float32")
    xs = []
    for i in range(len(values) - window):
        xs.append(values[i : i + window])
    if not xs:
        return None, None, None
    x_array = np.array(xs)
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1], 1))
    y_reg_all = df["Target_Reg"].values.astype("float32")
    y_class_all = df["Target_Class"].values
    y_reg_seq = y_reg_all[window - 1 :]
    y_class_seq = y_class_all[window - 1 :]
    min_len = min(len(x_array), len(y_reg_seq), len(y_class_seq))
    x_array = x_array[:min_len]
    y_reg_seq = y_reg_seq[:min_len]
    y_class_seq = y_class_seq[:min_len]
    return x_array, y_reg_seq, y_class_seq


def train_lstm_models(df: pd.DataFrame, window: int = 20):
    if not TF_AVAILABLE:
        return {}
    x_seq, y_reg_seq, y_class_seq_raw = build_lstm_sequences(df, window)
    if x_seq is None:
        return {}
    y_class_seq = y_class_seq_raw + 1
    split_seq = int(len(x_seq) * 0.8)
    x_train, x_test = x_seq[:split_seq], x_seq[split_seq:]
    y_reg_train, y_reg_test = y_reg_seq[:split_seq], y_reg_seq[split_seq:]
    y_class_train, y_class_test = y_class_seq[:split_seq], y_class_seq[split_seq:]
    idx_all = df.index[window - 1 : window - 1 + len(x_seq)]
    lstm_test_index = idx_all[split_seq:]
    reg_model = Sequential(
        [
            LSTM(32, input_shape=(window, 1)),
            Dense(1),
        ]
    )
    reg_model.compile(optimizer="adam", loss="mse")
    reg_model.fit(x_train, y_reg_train, epochs=5, batch_size=32, verbose=0)
    class_model = Sequential(
        [
            LSTM(32, input_shape=(window, 1)),
            Dense(3, activation="softmax"),
        ]
    )
    class_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    class_model.fit(x_train, y_class_train, epochs=5, batch_size=32, verbose=0)
    lstm_reg_pred_test = reg_model.predict(x_test, verbose=0).flatten()
    lstm_class_proba_test = class_model.predict(x_test, verbose=0)
    lstm_class_pred_test_idx = lstm_class_proba_test.argmax(axis=1)
    lstm_class_pred_test = lstm_class_pred_test_idx - 1
    return {
        "lstm_window": window,
        "lstm_X_test": x_test,
        "lstm_y_reg_test": y_reg_test,
        "lstm_y_class_test": y_class_test - 1,
        "lstm_test_index": lstm_test_index,
        "lstm_reg_model": reg_model,
        "lstm_class_model": class_model,
        "lstm_reg_pred_test": lstm_reg_pred_test,
        "lstm_class_pred_test": lstm_class_pred_test,
        "lstm_class_proba_test": lstm_class_proba_test,
    }


def train_models(df: pd.DataFrame):
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "Future_Price_7d",
            "Future_Return_7d",
            "Target_Class",
            "Target_Reg",
        ]
        and df[c].dtype != "O"
    ]
    data = df[feature_cols].values
    y_class_raw = df["Target_Class"].values
    y_class = y_class_raw + 1
    y_reg = df["Target_Reg"].values
    split = int(len(df) * 0.8)
    x_train, x_test = data[:split], data[split:]
    y_class_train, y_class_test = y_class[:split], y_class[split:]
    y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
    if LGB_AVAILABLE:
        class_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            random_state=42,
        )
        class_model.fit(x_train, y_class_train)
        reg_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="regression",
            random_state=42,
        )
        reg_model.fit(x_train, y_reg_train)
    else:
        class_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        class_model.fit(x_train, y_class_train)
        reg_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        reg_model.fit(x_train, y_reg_train)

    scaler_svc = StandardScaler()
    x_train_scaled = scaler_svc.fit_transform(x_train)
    x_test_scaled = scaler_svc.transform(x_test)

    svc_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svc_model.fit(x_train_scaled, y_class_train)

    class_proba_test = class_model.predict_proba(x_test)
    class_pred_test_idx = class_proba_test.argmax(axis=1)
    class_pred_test = class_pred_test_idx - 1
    max_conf_lgbm = class_proba_test.max(axis=1)
    class_pred_test = np.where(max_conf_lgbm < 0.55, 0, class_pred_test)

    svc_proba_test = svc_model.predict_proba(x_test_scaled)
    svc_pred_test_idx = svc_proba_test.argmax(axis=1)
    svc_pred_test_raw = svc_pred_test_idx - 1
    max_conf_svc = svc_proba_test.max(axis=1)
    svc_pred_test = np.where(max_conf_svc < 0.55, 0, svc_pred_test_raw)

    def _smooth_classes(pred, window: int = 3):
        arr = np.asarray(pred, dtype=int)
        if len(arr) <= window:
            return arr
        out = arr.copy()
        for i in range(window - 1, len(arr)):
            segment = arr[i - window + 1 : i + 1]
            vals, counts = np.unique(segment, return_counts=True)
            out[i] = vals[counts.argmax()]
        return out

    svc_pred_test = _smooth_classes(svc_pred_test, window=3)

    reg_pred_test = reg_model.predict(x_test)

    def _regression_metrics(y_true, y_pred):
        y_true_arr = np.asarray(y_true, dtype="float32")
        y_pred_arr = np.asarray(y_pred, dtype="float32")
        err = y_pred_arr - y_true_arr
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err**2))
        denom = np.where(np.abs(y_true_arr) < 1e-9, 1e-9, np.abs(y_true_arr))
        mape = float(np.mean(np.abs(err) / denom) * 100.0)
        return {"mae": mae, "mse": mse, "mape": mape}

    result = {
        "feature_cols": feature_cols,
        "x_test": x_test,
        "y_class_test": y_class_test - 1,
        "y_reg_test": y_reg_test,
        "class_model": class_model,
        "reg_model": reg_model,
        "class_pred_test": class_pred_test,
        "class_proba_test": class_proba_test,
        "reg_pred_test": reg_pred_test,
        "svc_model": svc_model,
        "svc_scaler": scaler_svc,
        "svc_pred_test": svc_pred_test,
        "svc_proba_test": svc_proba_test,
        "df": df,
        "split": split,
        "best_classifier": "svc",
    }
    lstm_data = train_lstm_models(df)
    if lstm_data:
        result.update(lstm_data)
        l_len = min(len(result["reg_pred_test"]), len(result["lstm_reg_pred_test"]))
        if l_len > 0:
            y_true_seg = result["y_reg_test"][-l_len:]
            lgbm_seg = result["reg_pred_test"][-l_len:]
            lstm_seg = result["lstm_reg_pred_test"][-l_len:]
            metrics_lgbm = _regression_metrics(y_true_seg, lgbm_seg)
            metrics_lstm = _regression_metrics(y_true_seg, lstm_seg)
            w_lgbm = 1.0 / (metrics_lgbm["mape"] + 1e-3)
            w_lstm = 1.0 / (metrics_lstm["mape"] + 1e-3)
            w_sum = w_lgbm + w_lstm
            w_lgbm /= w_sum
            w_lstm /= w_sum
            hybrid_reg = w_lgbm * lgbm_seg + w_lstm * lstm_seg
            result["hybrid_reg_pred_test"] = hybrid_reg

            lgbm_proba_last = result["class_proba_test"][-l_len:]
            lstm_proba_last = result["lstm_class_proba_test"][-l_len:]
            hybrid_proba = w_lgbm * lgbm_proba_last + w_lstm * lstm_proba_last
            hybrid_idx = hybrid_proba.argmax(axis=1)
            hybrid_class = hybrid_idx - 1
            max_conf_hybrid = hybrid_proba.max(axis=1)
            hybrid_class = np.where(max_conf_hybrid < 0.5, 0, hybrid_class)
            result["hybrid_class_pred_test"] = hybrid_class

            y_true_cls_lgbm = result["y_class_test"]
            acc_lgbm = float(
                np.mean(y_true_cls_lgbm == result["class_pred_test"])
            )
            y_true_cls_lstm = result["lstm_y_class_test"]
            acc_lstm = float(
                np.mean(y_true_cls_lstm == result["lstm_class_pred_test"])
            )
            y_true_cls_hybrid = result["y_class_test"][-l_len:]
            acc_hybrid = float(
                np.mean(y_true_cls_hybrid == hybrid_class)
            )
            y_true_cls_svc = result["y_class_test"]
            acc_svc = float(
                np.mean(y_true_cls_svc == svc_pred_test)
            )

            metrics_hybrid = _regression_metrics(y_true_seg, hybrid_reg)
            metrics_lgbm["acc"] = acc_lgbm
            metrics_lstm["acc"] = acc_lstm
            metrics_hybrid["acc"] = acc_hybrid
            metrics_svc = {"acc": acc_svc}

            result["metrics"] = {
                "lgbm": metrics_lgbm,
                "lstm": metrics_lstm,
                "hybrid": metrics_hybrid,
                "svc": metrics_svc,
                "weights": {"lgbm": float(w_lgbm), "lstm": float(w_lstm)},
            }
            best_name, best_acc = max(
                [
                    ("lgbm", acc_lgbm),
                    ("lstm", acc_lstm),
                    ("hybrid", acc_hybrid),
                    ("svc", acc_svc),
                ],
                key=lambda x: x[1],
            )
            result["best_by_accuracy"] = {
                "name": best_name,
                "acc": float(best_acc),
            }
    return result


def fetch_external_news(keyword: str, limit: int = 20):
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []
    url = "https://financialmodelingprep.com/api/v3/stock_news"
    params = {
        "tickers": keyword,
        "limit": limit,
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    items = []
    now = datetime.utcnow()
    for n in data:
        title = n.get("title", "")
        text = n.get("text", "")
        publisher = n.get("site", "")
        link = n.get("url", "")
        published = n.get("publishedDate")
        if isinstance(published, str):
            try:
                t_time = datetime.fromisoformat(published.replace(" ", "T"))
            except Exception:
                t_time = now
        else:
            t_time = now
        items.append(
            {
                "title": title,
                "summary": text,
                "publisher": publisher,
                "link": link,
                "time": t_time,
            }
        )
    return items


def fetch_fx_rss_news(limit: int = 20):
    url = "https://www.fxstreet.com/rss/news"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        content = resp.content
    except Exception:
        return []
    try:
        root = ET.fromstring(content)
    except Exception:
        return []
    items = []
    now = datetime.utcnow()
    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        pub_el = item.find("pubDate")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        summary = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        published_raw = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
        if published_raw:
            try:
                t_time = datetime.strptime(
                    published_raw[:25], "%a, %d %b %Y %H:%M:%S"
                )
            except Exception:
                t_time = now
        else:
            t_time = now
        items.append(
            {
                "title": title,
                "summary": summary,
                "publisher": "FXStreet",
                "link": link,
                "time": t_time,
            }
        )
        if len(items) >= limit:
            break
    return items


def fetch_twitter_news(keyword: str, limit: int = 20):
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        return []
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": keyword,
        "max_results": max(10, min(limit, 100)),
        "tweet.fields": "created_at,lang,author_id",
    }
    headers = {"Authorization": f"Bearer {bearer}"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    tweets = data.get("data") or []
    now = datetime.utcnow()
    items = []
    for tw in tweets[:limit]:
        text = tw.get("text", "")
        tid = tw.get("id")
        created_at = tw.get("created_at")
        if isinstance(created_at, str):
            try:
                t_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                t_time = now
        else:
            t_time = now
        link = f"https://twitter.com/i/web/status/{tid}" if tid else ""
        items.append(
            {
                "title": text,
                "summary": "",
                "publisher": "Twitter",
                "link": link,
                "time": t_time,
            }
        )
    return items


def fetch_crypto_rss_news(limit: int = 20):
    feeds = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ]
    items = []
    now = datetime.utcnow()
    for name, url in feeds:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            content = resp.content
        except Exception:
            continue
        try:
            root = ET.fromstring(content)
        except Exception:
            continue
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            summary = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
            published_raw = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
            if published_raw:
                try:
                    t_time = datetime.strptime(
                        published_raw[:25], "%a, %d %b %Y %H:%M:%S"
                    )
                except Exception:
                    t_time = now
            else:
                t_time = now
            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "publisher": name,
                    "link": link,
                    "time": t_time,
                }
            )
            if len(items) >= limit:
                return items
    return items


def fetch_official_rss_news(limit: int = 20):
    feeds = [
        ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
        ("ECB", "https://www.ecb.europa.eu/press/pr/date/rss.en.html"),
    ]
    items = []
    now = datetime.utcnow()
    for name, url in feeds:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            content = resp.content
        except Exception:
            continue
        try:
            root = ET.fromstring(content)
        except Exception:
            continue
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            summary = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
            published_raw = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
            if published_raw:
                try:
                    t_time = datetime.strptime(
                        published_raw[:25], "%a, %d %b %Y %H:%M:%S"
                    )
                except Exception:
                    t_time = now
            else:
                t_time = now
            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "publisher": name,
                    "link": link,
                    "time": t_time,
                }
            )
            if len(items) >= limit:
                return items
    return items


def compute_news_sentiment(ticker: str, instrument_name: str, limit: int = 20):
    candidates = [ticker, "EUR=X", "DX-Y.NYB"]
    now = datetime.now()
    raw_news = []
    for sym in candidates:
        try:
            t = yf.Ticker(sym)
            batch = t.news or []
        except Exception:
            batch = []
        if not batch:
            continue
        for item in batch:
            ts = item.get("providerPublishTime", None)
            if ts is not None:
                t_time = datetime.fromtimestamp(ts)
            else:
                t_time = now
            raw_news.append(
                {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "time": t_time,
                    "summary": "",
                }
            )
        if len(raw_news) >= limit:
            break
    twitter_news = fetch_twitter_news(instrument_name, limit=limit)
    if twitter_news:
        raw_news.extend(twitter_news)
    official_news = fetch_official_rss_news(limit=limit)
    if official_news:
        raw_news.extend(official_news)
    is_fx = ticker.endswith("=X") or "/" in instrument_name
    if is_fx:
        fx_news = fetch_fx_rss_news(limit=limit)
        if fx_news:
            raw_news.extend(fx_news)
    is_crypto = "BTC" in ticker.upper() or "BTC" in instrument_name.upper()
    if is_crypto:
        crypto_news = fetch_crypto_rss_news(limit=limit)
        if crypto_news:
            raw_news.extend(crypto_news)
    if not raw_news:
        items = [
            {
                "title": "Новости недоступны для данного тикера сейчас",
                "publisher": "system",
                "link": "",
                "time": now,
                "score": 0,
                "importance": "нет данных",
                "effect": f"нейтрально для {instrument_name}",
            }
        ]
        return items, 0.0
    raw_news.sort(key=lambda n: n.get("time") or now, reverse=True)
    raw_news = raw_news[:limit]
    positives = ["growth", "rise", "strong", "higher", "bull", "record", "gain"]
    negatives = ["fall", "drop", "weak", "lower", "bear", "loss", "risk"]
    macro_words = [
        "fed",
        "ecb",
        "rate",
        "rates",
        "hike",
        "cut",
        "inflation",
        "cpi",
        "gdp",
        "jobs",
        "payrolls",
    ]
    items = []
    total_score = 0
    total_count = 0
    official_score = 0
    official_count = 0
    for src in raw_news:
        title = src.get("title", "")
        summary = src.get("summary", "")
        text_lower = f"{title} {summary}".lower()
        score = 0
        if any(w in text_lower for w in positives):
            score += 1
        if any(w in text_lower for w in negatives):
            score -= 1
        t_time = src.get("time") or now
        age_hours = (now - t_time).total_seconds() / 3600.0
        importance = "низкая"
        if age_hours <= 24 and any(w in text_lower for w in macro_words):
            importance = "высокая"
        elif age_hours <= 72:
            importance = "средняя"
        usd_in_text = "usd" in text_lower or "dollar" in text_lower
        eur_in_text = "eur" in text_lower or "euro" in text_lower
        if score > 0 and eur_in_text:
            effect = f"поддерживает рост {instrument_name}"
        elif score > 0 and usd_in_text:
            effect = f"давит на {instrument_name}"
        elif score < 0 and eur_in_text:
            effect = f"давит на {instrument_name}"
        elif score < 0 and usd_in_text:
            effect = f"поддерживает рост {instrument_name}"
        else:
            effect = f"нейтрально для {instrument_name}"
        total_score += score
        total_count += 1
        if src.get("publisher") in ("Federal Reserve", "ECB"):
            official_score += score
            official_count += 1
        items.append(
            {
                "title": title,
                "publisher": src.get("publisher", ""),
                "link": src.get("link", ""),
                "time": t_time,
                "score": score,
                "importance": importance,
                "effect": effect,
                "summary": summary,
            }
        )
    avg_official = official_score / official_count if official_count else 0.0
    return items, avg_official


def fetch_future_economic_events(days_ahead: int = 7):
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []
    start = datetime.utcnow().date()
    end = start + timedelta(days=days_ahead)
    url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    events = []
    for ev in data:
        country = (ev.get("country") or "").upper()
        if country not in ("US", "USA", "EU", "EMU", "DE", "FR"):
            continue
        date_str = ev.get("date") or ev.get("event_date")
        try:
            when = datetime.fromisoformat(date_str)
        except Exception:
            when = None
        event_name = ev.get("event") or ev.get("title") or ""
        impact = ev.get("impact") or ev.get("importance") or ""
        actual = ev.get("actual")
        previous = ev.get("previous")
        estimate = ev.get("estimate")
        events.append(
            {
                "time": when,
                "country": country,
                "event": event_name,
                "impact": impact,
                "actual": actual,
                "previous": previous,
                "estimate": estimate,
            }
        )
    events.sort(
        key=lambda e: (
            e["time"] or datetime.max,
            {"High": 2, "Medium": 1}.get(str(e["impact"]).title(), 0),
        )
    )
    return events


def score_future_events(events) -> float:
    if not events:
        return 0.0
    total = 0.0
    for e in events:
        impact_raw = str(e.get("impact", "") or "")
        impact = impact_raw.lower()
        if "high" in impact or "высок" in impact:
            w = 2.0
        elif "medium" in impact or "средн" in impact:
            w = 1.0
        else:
            w = 0.5
        country = (e.get("country") or "").upper()
        if country in ("EU", "EMU", "DE", "FR"):
            total += w
        elif country in ("US", "USA"):
            total -= w
    return float(np.tanh(total / 5.0))


def detect_patterns(prices: pd.Series):
    if len(prices) < 60:
        return []
    data = prices.values
    patterns = []
    last = data[-60:]
    z = (last - last.mean()) / (last.std() + 1e-9)
    if z[-1] > 0.5 and z[-2] > 0.5 and z[-3] < 0.0:
        patterns.append("Возможный разворот вверх (локальный минимум)")
    if z[-1] < -0.5 and z[-2] < -0.5 and z[-3] > 0.0:
        patterns.append("Возможный разворот вниз (локальный максимум)")
    diffs = np.diff(last)
    up_ratio = (diffs > 0).mean()
    if up_ratio > 0.7:
        patterns.append("Устойчивый восходящий тренд")
    if up_ratio < 0.3:
        patterns.append("Устойчивый нисходящий тренд")
    recent = last[-40:]
    base_trend = recent[-1] - recent[0]
    recent_std = np.std(recent)
    mid = recent[:20]
    tail = recent[20:]
    if abs(base_trend) > 0.03 * mid[0] and np.std(tail) < recent_std * 0.7:
        if base_trend > 0:
            patterns.append("Возможная фигура флаг после роста")
        else:
            patterns.append("Возможная фигура флаг после падения")
    if abs(base_trend) > 0.03 * mid[0]:
        tail_high = tail.max()
        tail_low = tail.min()
        if (tail_high - tail_low) < 0.01 * last.mean():
            if base_trend > 0:
                patterns.append("Возможная фигура вымпел после роста")
            else:
                patterns.append("Возможная фигура вымпел после падения")
    local_max = []
    local_min = []
    for i in range(1, len(last) - 1):
        if last[i] > last[i - 1] and last[i] > last[i + 1]:
            local_max.append((i, last[i]))
        if last[i] < last[i - 1] and last[i] < last[i + 1]:
            local_min.append((i, last[i]))
    if len(local_max) >= 3:
        p1, p2, p3 = local_max[-3], local_max[-2], local_max[-1]
        h1 = p1[1]
        h2 = p2[1]
        h3 = p3[1]
        head = max(h1, h2, h3)
        shoulders = [h for h in (h1, h3) if h < head]
        if len(shoulders) == 2:
            if abs(shoulders[0] - shoulders[1]) / head < 0.02 and head - max(shoulders) > 0.01 * head:
                patterns.append("Возможная фигура голова и плечи")
    if len(local_max) >= 2:
        a, b = local_max[-2], local_max[-1]
        h1, h2 = a[1], b[1]
        if abs(h1 - h2) / max(h1, h2) < 0.005:
            left = min(a[0], b[0])
            right = max(a[0], b[0])
            valley = last[left:right].min()
            if valley < min(h1, h2) * 0.995:
                patterns.append("Возможная фигура двойная вершина")
    if len(local_min) >= 2:
        a, b = local_min[-2], local_min[-1]
        l1, l2 = a[1], b[1]
        if abs(l1 - l2) / min(l1, l2) < 0.005:
            left = min(a[0], b[0])
            right = max(a[0], b[0])
            peak = last[left:right].max()
            if peak > max(l1, l2) * 1.005:
                patterns.append("Возможная фигура двойное дно")
    if not patterns:
        patterns.append("Явно выраженных фигур не обнаружено")
    return patterns


def build_price_chart(df: pd.DataFrame, instrument_name: str, patterns=None) -> go.Figure:
    df_plot = df.tail(300)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["Close"],
            mode="lines",
            name="Close",
            line=dict(color="black", width=2),
        )
    )
    for span, color in [(8, "blue"), (21, "orange"), (55, "green")]:
        col = f"EMA_{span}"
        if col in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot.index,
                    y=df_plot[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.5),
                )
            )
    if not df_plot.empty:
        last_date = df_plot.index[-1]
        last_price = float(df_plot["Close"].iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[last_price],
                mode="markers",
                name="Сегодня",
                marker=dict(color="black", size=9, symbol="diamond"),
            )
        )
        fig.add_vline(
            x=last_date,
            line_color="gray",
            line_dash="dot",
            opacity=0.6,
        )
        if patterns:
            text = "; ".join(patterns[:2])
            fig.add_annotation(
                x=last_date,
                y=last_price,
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=10),
            )
    fig.update_layout(
        title=f"{instrument_name}: цена и EMA",
        xaxis_title="Дата",
        yaxis_title="Цена",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def build_atr_chart(df: pd.DataFrame, instrument_name: str) -> go.Figure:
    df_plot = df.tail(300)
    fig = go.Figure()
    if "ATR_14" in df_plot.columns:
        atr_series = df_plot["ATR_14"].dropna()
        if not atr_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=atr_series.index,
                    y=atr_series.values,
                    mode="lines",
                    name="ATR(14)",
                    line=dict(color="purple", width=2),
                )
            )
            last_date = atr_series.index[-1]
            last_value = float(atr_series.iloc[-1])
            fig.add_trace(
                go.Scatter(
                    x=[last_date],
                    y=[last_value],
                    mode="markers",
                    name="Текущее значение ATR(14)",
                    marker=dict(color="black", size=9, symbol="diamond"),
                )
            )
            fig.add_vline(
                x=last_date,
                line_color="gray",
                line_dash="dot",
                opacity=0.6,
            )
    fig.update_layout(
        title=f"{instrument_name}: ATR(14) и волатильность",
        xaxis_title="Дата",
        yaxis_title="ATR(14)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def build_prediction_chart(
    df: pd.DataFrame,
    model_data,
    price_model: str = "lstm",
    model_weights: dict | None = None,
) -> go.Figure:
    lstm_idx = model_data.get("lstm_test_index")
    lstm_true = model_data.get("lstm_y_reg_test")
    lstm_pred = model_data.get("lstm_reg_pred_test")

    if lstm_idx is not None and lstm_true is not None and lstm_pred is not None:
        dates = lstm_idx
        true_prices = lstm_true
        pred_prices = lstm_pred
    else:
        split = model_data["split"]
        df_tail = df.iloc[split:]
        dates = df_tail.index
        true_prices = model_data["y_reg_test"]
        pred_prices = model_data["reg_pred_test"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=true_prices,
            mode="lines",
            name="Факт",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred_prices,
            mode="lines",
            name="Прогноз модели",
            line=dict(color="green", width=2, dash="dot"),
        )
    )
    if len(dates) > 0:
        last_date = dates[-1]
        last_true = float(true_prices[-1])
        fig.add_trace(
            go.Scatter(
                x=[last_date],
                y=[last_true],
                mode="markers",
                name="Последняя точка",
                marker=dict(color="black", size=9, symbol="x"),
            )
        )
    if len(df) >= 1:
        last_row = df.iloc[-1]
        feature_cols = model_data["feature_cols"]
        x_last = last_row[feature_cols].values.reshape(1, -1)
        reg_model = model_data["reg_model"]
        last_reg_price_lgbm = float(reg_model.predict(x_last)[0])
        window = model_data.get("lstm_window", 20)
        close_series = df["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        values = close_series.values.astype("float32")
        has_lstm = len(values) >= window and "lstm_reg_model" in model_data
        last_reg_price_lstm = None
        last_reg_price_hybrid = None
        if has_lstm:
            seq = values[-window:]
            seq = seq.reshape((1, window, 1))
            lstm_reg_model = model_data["lstm_reg_model"]
            last_reg_price_lstm = float(
                lstm_reg_model.predict(seq, verbose=0)[0, 0]
            )
            if (
                model_weights
                and "lgbm" in model_weights
                and "lstm" in model_weights
            ):
                w_lgbm = float(model_weights["lgbm"])
                w_lstm = float(model_weights["lstm"])
            else:
                weights_auto = (
                    model_data.get("metrics", {}).get("weights")
                    if model_data.get("metrics")
                    else None
                )
                if weights_auto:
                    w_lgbm = float(weights_auto.get("lgbm", 0.5))
                    w_lstm = float(weights_auto.get("lstm", 0.5))
                else:
                    w_lgbm = 0.5
                    w_lstm = 0.5
            last_reg_price_hybrid = (
                w_lgbm * last_reg_price_lgbm + w_lstm * last_reg_price_lstm
            )
        price_model_lower = (price_model or "lstm").lower()
        if price_model_lower == "hybrid" and last_reg_price_hybrid is not None:
            forecast_price = last_reg_price_hybrid
        elif has_lstm and last_reg_price_lstm is not None:
            forecast_price = last_reg_price_lstm
        else:
            forecast_price = last_reg_price_lgbm
        last_date_price = df.index[-1]
        last_price = float(df["Close"].iloc[-1])
        if len(df.index) >= 2:
            step = df.index[-1] - df.index[-2]
        else:
            step = timedelta(days=7)
        forecast_date = last_date_price + step * 7
        fig.add_trace(
            go.Scatter(
                x=[last_date_price, forecast_date],
                y=[last_price, forecast_price],
                mode="lines+markers",
                name="Прогноз на 7 шагов вперед",
                line=dict(color="purple", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        title="Прогноз цены через 7 дней",
        xaxis_title="Дата",
        yaxis_title="Цена",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def build_lstm_chart(df: pd.DataFrame, model_data) -> go.Figure:
    idx = model_data.get("lstm_test_index")
    y_true = model_data.get("lstm_y_reg_test")
    y_pred = model_data.get("lstm_reg_pred_test")
    if idx is None or y_true is None or y_pred is None:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y_true,
            mode="lines",
            name="Факт",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=y_pred,
            mode="lines",
            name="Прогноз LSTM",
            line=dict(color="green", width=2, dash="dot"),
        )
    )
    if len(idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=[idx[-1]],
                y=[float(y_pred[-1])],
                mode="markers",
                name="Последний прогноз LSTM",
                marker=dict(color="green", size=9, symbol="x"),
            )
        )
    fig.update_layout(
        title="Отдельный прогноз LSTM (цена через 7 дней)",
        xaxis_title="Дата",
        yaxis_title="Цена",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def build_classification_chart(df: pd.DataFrame, model_data, classifier_override: str | None = None) -> go.Figure:
    split = model_data["split"]
    df_tail = df.iloc[split:]
    dates = df_tail.index
    prices_base = df_tail["Close"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices_base,
            mode="lines",
            name="Close",
            line=dict(color="black", width=1.5),
        )
    )

    def add_signals(pred, idx, prices, prefix: str, opacity: float):
        if pred is None or len(pred) == 0:
            return
        pred = np.asarray(pred)
        idx = pd.Index(idx)
        prices_series = pd.Series(prices, index=idx)
        if len(pred) != len(idx):
            l_len = min(len(pred), len(idx))
            pred = pred[-l_len:]
            idx = idx[-l_len:]
            prices_series = prices_series[-l_len:]
        for cls, name, color in [(-1, "SELL", "red"), (0, "HOLD", "gray"), (1, "BUY", "green")]:
            mask = pred == cls
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=idx[mask],
                    y=prices_series[mask],
                    mode="markers",
                    name=f"{prefix} {name}",
                    marker=dict(color=color, size=9 if prefix == "HYBRID" else 7),
                    opacity=opacity,
                )
            )

    has_lstm = (
        model_data.get("lstm_class_pred_test") is not None
        and model_data.get("lstm_test_index") is not None
    )
    has_hybrid = model_data.get("hybrid_class_pred_test") is not None
    has_lgbm = model_data.get("class_pred_test") is not None
    has_svc = model_data.get("svc_pred_test") is not None

    requested_cls = classifier_override or model_data.get("best_classifier", "lgbm")
    best_cls = requested_cls

    if best_cls == "lstm" and not has_lstm:
        best_cls = "hybrid" if has_hybrid else ("lgbm" if has_lgbm else ("svc" if has_svc else None))
    elif best_cls == "hybrid" and not has_hybrid:
        best_cls = "lstm" if has_lstm else ("lgbm" if has_lgbm else ("svc" if has_svc else None))
    elif best_cls == "lgbm" and not has_lgbm:
        best_cls = "hybrid" if has_hybrid else ("lstm" if has_lstm else ("svc" if has_svc else None))
    elif best_cls == "svc" and not has_svc:
        best_cls = "hybrid" if has_hybrid else ("lstm" if has_lstm else ("lgbm" if has_lgbm else None))

    if best_cls == "lstm" and has_lstm:
        lstm_pred = model_data.get("lstm_class_pred_test")
        lstm_idx = model_data.get("lstm_test_index")
        lstm_idx = pd.Index(lstm_idx)
        prices_lstm = df["Close"].reindex(lstm_idx)
        add_signals(lstm_pred, lstm_idx, prices_lstm, "LSTM", 1.0)
    elif best_cls == "hybrid" and has_hybrid:
        hybrid_pred = model_data.get("hybrid_class_pred_test")
        add_signals(hybrid_pred, dates, prices_base, "HYBRID", 1.0)
    elif best_cls == "lgbm" and has_lgbm:
        lgbm_pred = model_data.get("class_pred_test")
        add_signals(lgbm_pred, dates, prices_base, "LGBM", 1.0)
    elif best_cls == "svc" and has_svc:
        svc_pred = model_data.get("svc_pred_test")
        add_signals(svc_pred, dates, prices_base, "SVC", 1.0)

    fig.update_layout(
        title="Сигналы классификации (BUY/SELL/HOLD)",
        xaxis_title="Дата",
        yaxis_title="Цена",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def build_classification_comparison_chart(
    df: pd.DataFrame,
    model_data,
    model_weights: dict | None = None,
    cls_conf_threshold: float | None = None,
) -> go.Figure:
    split = model_data["split"]
    df_tail = df.iloc[split:]
    dates = df_tail.index
    prices_base = df_tail["Close"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices_base,
            mode="lines",
            name="Close",
            line=dict(color="black", width=1.0),
        )
    )

    def add_signals(pred, idx, prices, prefix: str, opacity: float):
        if pred is None or len(pred) == 0:
            return
        pred = np.asarray(pred)
        idx = pd.Index(idx)
        prices_series = pd.Series(prices, index=idx)
        if len(pred) != len(idx):
            l_len = min(len(pred), len(idx))
            pred = pred[-l_len:]
            idx = idx[-l_len:]
            prices_series = prices_series[-l_len:]
        for cls, name, color in [(-1, "SELL", "red"), (0, "HOLD", "gray"), (1, "BUY", "green")]:
            mask = pred == cls
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=idx[mask],
                    y=prices_series[mask],
                    mode="markers",
                    name=f"{prefix} {name}",
                    marker=dict(
                        color=color,
                        size=8 if prefix == "HYBRID" else 6,
                        symbol="circle",
                    ),
                    opacity=opacity,
                )
            )

    metrics = model_data.get("metrics") or {}
    weights_auto = metrics.get("weights") or {}
    if model_weights and "lgbm" in model_weights and "lstm" in model_weights:
        w_lgbm = float(model_weights["lgbm"])
        w_lstm = float(model_weights["lstm"])
        has_weights = True
    elif "lgbm" in weights_auto and "lstm" in weights_auto:
        w_lgbm = float(weights_auto["lgbm"])
        w_lstm = float(weights_auto["lstm"])
        has_weights = True
    else:
        w_lgbm = 0.5
        w_lstm = 0.5
        has_weights = False

    if has_weights and cls_conf_threshold is not None:
        hybrid_prefix = f"Hybrid (wLGBM={w_lgbm:.2f}, thr={cls_conf_threshold:.2f})"
    elif has_weights:
        hybrid_prefix = f"Hybrid (wLGBM={w_lgbm:.2f})"
    else:
        hybrid_prefix = "Hybrid"

    lgbm_pred = model_data.get("class_pred_test")
    if lgbm_pred is not None:
        add_signals(lgbm_pred, dates, prices_base, "LGBM", 0.7)

    hybrid_pred = model_data.get("hybrid_class_pred_test")
    if hybrid_pred is not None:
        add_signals(hybrid_pred, dates, prices_base, hybrid_prefix, 1.0)

    lstm_pred = model_data.get("lstm_class_pred_test")
    lstm_idx = model_data.get("lstm_test_index")
    if lstm_pred is not None and lstm_idx is not None:
        lstm_idx = pd.Index(lstm_idx)
        prices_lstm = df["Close"].reindex(lstm_idx)
        add_signals(lstm_pred, lstm_idx, prices_lstm, "LSTM", 0.7)

    svc_pred = model_data.get("svc_pred_test")
    if svc_pred is not None:
        add_signals(svc_pred, dates, prices_base, "SVC", 0.7)

    fig.update_layout(
        title="Сравнение сигналов LGBM / LSTM / Hybrid (BUY/SELL/HOLD)",
        xaxis_title="Дата",
        yaxis_title="Цена",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def combine_signals(
    df: pd.DataFrame,
    model_data,
    news_score: float,
    patterns: list,
    future_events_score: float,
    model_weights: dict | None,
    price_model: str,
    classifier_override: str | None = None,
    cls_conf_threshold: float = 0.55,
):
    last_row = df.iloc[-1]
    feature_cols = model_data["feature_cols"]
    x_last = last_row[feature_cols].values.reshape(1, -1)
    reg_model = model_data["reg_model"]
    class_model = model_data.get("class_model")
    svc_model = model_data.get("svc_model")
    svc_scaler = model_data.get("svc_scaler")
    last_price = float(last_row["Close"])
    last_reg_price_lgbm = float(reg_model.predict(x_last)[0])

    window = model_data.get("lstm_window", 20)
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    values = close_series.values.astype("float32")

    has_lstm = (
        len(values) >= window
        and "lstm_reg_model" in model_data
    )
    last_reg_price_lstm = None
    last_reg_price_hybrid = None
    if has_lstm:
        seq = values[-window:]
        seq = seq.reshape((1, window, 1))
        lstm_reg_model = model_data["lstm_reg_model"]
        last_reg_price_lstm = float(lstm_reg_model.predict(seq, verbose=0)[0, 0])

        if model_weights and "lgbm" in model_weights and "lstm" in model_weights:
            w_lgbm = float(model_weights["lgbm"])
            w_lstm = float(model_weights["lstm"])
        else:
            weights_auto = (
                model_data.get("metrics", {}).get("weights")
                if model_data.get("metrics")
                else None
            )
            if weights_auto:
                w_lgbm = float(weights_auto.get("lgbm", 0.5))
                w_lstm = float(weights_auto.get("lstm", 0.5))
            else:
                w_lgbm = 0.5
                w_lstm = 0.5

        last_reg_price_hybrid = w_lgbm * last_reg_price_lgbm + w_lstm * last_reg_price_lstm

    best_cls = classifier_override or model_data.get("best_classifier", "hybrid")
    last_class = 0
    prob_max = None
    if best_cls == "svc" and svc_model is not None and svc_scaler is not None:
        k = min(3, len(df))
        recent_block = df.iloc[-k:][feature_cols].values
        recent_scaled = svc_scaler.transform(recent_block)
        proba_seq = svc_model.predict_proba(recent_scaled)
        proba = proba_seq.mean(axis=0)
        idx = int(proba.argmax())
        prob_max = float(proba[idx])
        last_class = idx - 1
    elif best_cls == "lgbm" and class_model is not None:
        proba = class_model.predict_proba(x_last)[0]
        idx = int(proba.argmax())
        prob_max = float(proba[idx])
        last_class = idx - 1
    elif best_cls in ("lstm", "hybrid") and has_lstm and "lstm_class_model" in model_data:
        lstm_class_model = model_data["lstm_class_model"]
        proba_lstm = lstm_class_model.predict(seq, verbose=0)[0]
        if best_cls == "lstm":
            proba = proba_lstm
        else:
            proba_lgbm = class_model.predict_proba(x_last)[0] if class_model is not None else proba_lstm
            proba = w_lgbm * proba_lgbm + w_lstm * proba_lstm
        idx = int(proba.argmax())
        prob_max = float(proba[idx])
        last_class = idx - 1

    if prob_max is not None and prob_max < cls_conf_threshold:
        last_class = 0

    price_model_lower = (price_model or "lstm").lower()
    if price_model_lower == "hybrid" and last_reg_price_hybrid is not None:
        last_reg_price = last_reg_price_hybrid
    elif has_lstm and last_reg_price_lstm is not None:
        last_reg_price = last_reg_price_lstm
    else:
        last_reg_price = last_reg_price_lgbm

    expected_return = (last_reg_price - last_price) / last_price
    pattern_bias = 0
    for p in patterns:
        if "восходящий" in p or "разворот вверх" in p:
            pattern_bias += 1
        if "нисходящий" in p or "разворот вниз" in p:
            pattern_bias -= 1

    class_component = last_class
    if prob_max is not None:
        class_component = last_class * prob_max

    score = (
        0.42 * class_component
        + 0.35 * np.tanh(expected_return * 50)
        + 0.10 * news_score
        + 0.08 * pattern_bias
        + 0.05 * future_events_score
    )

    if score > 0.4:
        action = "BUY"
    elif score < -0.4:
        action = "SELL"
    else:
        action = "HOLD"

    reasons = []
    classifier_label_map = {
        "svc": "SVC",
        "lgbm": "LGBM",
        "lstm": "LSTM",
        "hybrid": "Hybrid",
    }
    cls_label = classifier_label_map.get(best_cls, str(best_cls).upper())
    if last_class > 0:
        reasons.append(f"классификация ({cls_label}) склоняется к BUY")
    elif last_class < 0:
        reasons.append(f"классификация ({cls_label}) склоняется к SELL")
    if prob_max is not None:
        reasons.append(f"уверенность классификации: {prob_max*100:.1f}%")
    reg_name = "LSTM" if price_model_lower == "lstm" else "Hybrid"
    if expected_return > 0:
        reasons.append(f"регрессия ({reg_name}) ожидает рост цены")
    elif expected_return < 0:
        reasons.append(f"регрессия ({reg_name}) ожидает снижение цены")
    if news_score > 0.1:
        reasons.append("официальные новости в среднем поддерживают рост")
    elif news_score < -0.1:
        reasons.append("официальные новости в среднем поддерживают падение")
    if future_events_score > 0.1:
        reasons.append("будущий календарь в целом поддерживает рост")
    elif future_events_score < -0.1:
        reasons.append("будущий календарь в целом поддерживает падение")
    if pattern_bias > 0:
        reasons.append("паттерны указывают на восходящий сценарий")
    elif pattern_bias < 0:
        reasons.append("паттерны указывают на нисходящий сценарий")
    if not reasons:
        reasons.append("сигналы противоречивы, используется более нейтральный подход")

    reason_text = "; ".join(reasons)
    details = {
        "class_signal": last_class,
        "expected_return": expected_return,
        "news_score": news_score,
        "pattern_bias": pattern_bias,
        "score": score,
        "action": action,
        "last_price": last_price,
        "target_price": last_reg_price,
        "reason": reason_text,
        "cls_model_used": best_cls,
    }
    return details


def enrich_signals_with_atr(signals: dict, atr_info: dict | None):
    if not atr_info:
        return signals
    level = atr_info.get("level")
    current = atr_info.get("current")
    if current is None or level is None:
        return signals
    if level == "высокая":
        text = f"волатильность сейчас высокая по ATR(14) ({current:.5f})"
    elif level == "низкая":
        text = f"волатильность сейчас низкая по ATR(14) ({current:.5f})"
    else:
        text = f"волатильность сейчас средняя по ATR(14) ({current:.5f})"
    reason = signals.get("reason") or ""
    if reason:
        reason = reason + "; " + text
    else:
        reason = text
    signals["reason"] = reason
    signals["atr_14"] = float(current)
    signals["atr_level"] = level
    return signals


def get_signals_for_ticker(
    ticker: str,
    instrument_name: str,
    years: int = 5,
    interval: str = "1d",
    profile: str | None = None,
):
    settings_all = INSTRUMENT_SETTINGS.get(ticker, {})
    profiles = settings_all.get("profiles")
    if profiles:
        profile_names = list(profiles.keys())
        if profile and profile in profiles:
            selected_profile = profile
        else:
            default_profile_name = settings_all.get("default_profile") or profile_names[0]
            if default_profile_name in profiles:
                selected_profile = default_profile_name
            else:
                selected_profile = profile_names[0]
        profile_cfg = profiles[selected_profile]
    else:
        selected_profile = profile
        profile_cfg = settings_all if isinstance(settings_all, dict) else {}
    horizon = profile_cfg.get("horizon", 7)
    lower_q = profile_cfg.get("lower_q", 0.33)
    upper_q = profile_cfg.get("upper_q", 0.66)
    cls_conf_default = profile_cfg.get("cls_conf_default", 0.55)
    df_raw = load_price_data(ticker, years=years, interval=interval)
    if df_raw is None or df_raw.empty:
        raise ValueError("Не удалось загрузить данные по тикеру")
    df_full = add_features(df_raw)
    atr_info = get_atr_volatility_info(df_full)

    support_level = None
    resistance_level = None
    if "Low_20" in df_full.columns:
        try:
            val = float(df_full["Low_20"].iloc[-1])
            if not np.isnan(val):
                support_level = val
        except Exception:
            support_level = None
    if "High_20" in df_full.columns:
        try:
            val = float(df_full["High_20"].iloc[-1])
            if not np.isnan(val):
                resistance_level = val
        except Exception:
            resistance_level = None

    df_model = add_targets(
        df_full.copy(),
        horizon=horizon,
        lower_q=lower_q,
        upper_q=upper_q,
    )
    model_data = train_models(df_model)
    metrics = model_data.get("metrics")
    model_weights = None
    price_model = "lstm"
    if metrics:
        model_weights = metrics.get("weights")
        if "hybrid" in metrics:
            price_model = "hybrid"
    cls_conf_threshold = cls_conf_default
    news_items, news_score = compute_news_sentiment(ticker, instrument_name)
    future_events = fetch_future_economic_events(days_ahead=7)
    future_events_score = score_future_events(future_events)
    patterns = detect_patterns(df_full["Close"])
    signals = combine_signals(
        df_full,
        model_data,
        news_score,
        patterns,
        future_events_score,
        model_weights,
        price_model,
        None,
        cls_conf_threshold,
    )
    signals = enrich_signals_with_atr(signals, atr_info)
    signals["support_level"] = support_level
    signals["resistance_level"] = resistance_level
    result = {
        "ticker": ticker,
        "instrument_name": instrument_name,
        "profile": selected_profile,
        "years": years,
        "interval": interval,
        "horizon": horizon,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "news_score": float(news_score),
        "future_events_score": float(future_events_score),
        "future_events": future_events,
        "patterns": patterns,
        "signal": {
            "action": signals["action"],
            "score": float(signals["score"]),
            "class_signal": int(signals["class_signal"]),
            "expected_return": float(signals["expected_return"]),
            "last_price": float(signals["last_price"]),
            "target_price": float(signals["target_price"]),
            "pattern_bias": int(signals["pattern_bias"]),
            "reason": signals["reason"],
            "cls_model_used": signals.get("cls_model_used"),
            "atr_14": float(signals["atr_14"]) if signals.get("atr_14") is not None else None,
            "atr_level": signals.get("atr_level"),
            "support_level": support_level,
            "resistance_level": resistance_level,
        },
    }
    return result


def build_dashboard_for_ticker(ticker: str, instrument_name: str):
    st.title(f"AI-анализ {instrument_name}: цена, новости, паттерны и рекомендации")
    st.markdown("Данные, модели, новости Twitter/официальных источников и паттерны объединены в панель.")
    col_params, col_info = st.columns([2, 1])
    with col_params:
        years = st.slider("Глубина истории (лет)", 1, 10, 5)
        interval = st.selectbox("Таймфрейм", ["1d", "1h", "4h"])
    with col_info:
        st.write(f"Инструмент: {instrument_name}")
        st.write(f"Тикер: {ticker}")
    settings_all = INSTRUMENT_SETTINGS.get(ticker, {})
    profiles = settings_all.get("profiles")
    if profiles:
        profile_names = list(profiles.keys())
        default_profile_name = settings_all.get("default_profile") or profile_names[0]
        try:
            default_index = profile_names.index(default_profile_name)
        except ValueError:
            default_index = 0
        selected_profile = st.sidebar.selectbox(
            "Профиль таргета и классификации",
            profile_names,
            index=default_index,
            key=f"profile_{ticker}",
        )
        profile_cfg = profiles[selected_profile]
    else:
        selected_profile = None
        profile_cfg = settings_all
    horizon = profile_cfg.get("horizon", 7)
    lower_q = profile_cfg.get("lower_q", 0.33)
    upper_q = profile_cfg.get("upper_q", 0.66)
    cls_conf_default = profile_cfg.get("cls_conf_default", 0.55)
    with st.spinner("Загружаю и обрабатываю данные..."):
        df_raw = load_price_data(ticker, years=years, interval=interval)
        if df_raw is None or df_raw.empty:
            st.error(
                "Не удалось загрузить данные по тикеру. "
                "Попробуйте другой таймфрейм или проверьте интернет/Yahoo Finance."
            )
            st.stop()
        df_full = add_features(df_raw)
        atr_info = get_atr_volatility_info(df_full)
        df_model = add_targets(df_full.copy(), horizon=horizon, lower_q=lower_q, upper_q=upper_q)
        model_data = train_models(df_model)
    st.caption(
        f"Данные загружены за период: {df_full.index.min().date()} — {df_full.index.max().date()}"
    )
    metrics = model_data.get("metrics")
    st.sidebar.subheader("Дополнительные графики")
    show_lstm_chart = st.sidebar.checkbox("Показать отдельный график LSTM", value=False)
    price_model = "lstm"
    model_weights = None
    classifier_override = None
    cls_conf_threshold = cls_conf_default if "cls_conf_default" in locals() else 0.55
    cls_mode = "Авто (лучшая по accuracy)"
    if metrics:
        st.sidebar.subheader("Вес гибридной модели")
        use_manual_weights = st.sidebar.checkbox(
            "Ручная настройка весов LGBM/LSTM", value=False
        )
        base_w_lgbm = float(metrics.get("weights", {}).get("lgbm", 0.5))
        if use_manual_weights:
            w_lgbm_manual = st.sidebar.slider(
                "Вес LGBM в гибриде", 0.0, 1.0, base_w_lgbm, step=0.05
            )
            model_weights = {"lgbm": w_lgbm_manual, "lstm": 1.0 - w_lgbm_manual}
        else:
            model_weights = metrics.get("weights")
        if "hybrid" in metrics:
            choice = st.sidebar.selectbox(
                "Модель для прогноза цены (7 дней)",
                ["LSTM", "Hybrid (LGBM+LSTM)"],
            )
            price_model = "hybrid" if "Hybrid" in choice else "lstm"
        st.sidebar.caption(
            f"horizon: {horizon} шагов, квантили таргета: [{lower_q:.2f}, {upper_q:.2f}]"
        )
        st.sidebar.subheader("Модель классификации")
        cls_mode = st.sidebar.radio(
            "Режим",
            ["Авто (SVC как основная)", "Ручной выбор"],
            index=0,
        )
        best_by_acc = model_data.get("best_by_accuracy") or {}
        best_name_internal = best_by_acc.get("name")
        best_label_map = {
            "svc": "SVC",
            "lstm": "LSTM",
            "lgbm": "LGBM",
            "hybrid": "Hybrid",
        }
        best_label = (
            best_label_map.get(best_name_internal, str(best_name_internal).upper())
            if best_name_internal is not None
            else None
        )
        best_acc = best_by_acc.get("acc")
        if best_label is not None and best_acc is not None:
            st.sidebar.caption(
                f"Лучшая по accuracy: {best_label} (Accuracy {best_acc*100:.2f}%)"
            )
        st.sidebar.subheader("Параметры доверия классификации")
        cls_conf_threshold = st.sidebar.slider(
            "Порог уверенности (классификатор)",
            0.40,
            0.90,
            cls_conf_threshold,
            step=0.01,
        )
        if cls_mode == "Ручной выбор":
            cls_label = st.sidebar.selectbox(
                "Модель для BUY/SELL/HOLD",
                ["SVC", "LSTM", "Hybrid"],
            )
            mapping = {
                "SVC": "svc",
                "LSTM": "lstm",
                "Hybrid": "hybrid",
            }
            classifier_override = mapping.get(cls_label)
    news_items, news_score = compute_news_sentiment(ticker, instrument_name)
    future_events = fetch_future_economic_events(days_ahead=7)
    future_events_score = score_future_events(future_events)
    patterns = detect_patterns(df_full["Close"])
    price_fig = build_price_chart(df_full, instrument_name, patterns=patterns)
    atr_fig = build_atr_chart(df_full, instrument_name)
    pred_fig = build_prediction_chart(df_model, model_data, price_model, model_weights)
    class_fig = build_classification_chart(df_model, model_data, classifier_override)
    st.subheader("Графики цены и прогнозов")
    st.plotly_chart(price_fig, use_container_width=True)
    st.plotly_chart(pred_fig, use_container_width=True)
    st.subheader("ATR и волатильность")
    st.plotly_chart(atr_fig, use_container_width=True)
    if atr_info:
        level = atr_info.get("level")
        current = atr_info.get("current")
        if level == "высокая":
            vol_text = "Волатильность сейчас высокая"
        elif level == "низкая":
            vol_text = "Волатильность сейчас низкая"
        else:
            vol_text = "Волатильность сейчас средняя"
        st.write(f"{vol_text} (ATR(14) ≈ {current:.5f})")
    if metrics:
        st.markdown("**Качество моделей (тестовый отрезок, регрессия/классификация):**")
        col_lgbm, col_lstm, col_hybrid, col_svc = st.columns(4)
        if "lgbm" in metrics:
            m = metrics["lgbm"]
            with col_lgbm:
                st.markdown("LightGBM")
                st.write(f"MAPE: {m['mape']:.2f}%")
                st.write(f"MAE: {m['mae']:.5f}")
                st.write(f"MSE: {m['mse']:.6f}")
                if "acc" in m:
                    st.write(f"Accuracy (class): {m['acc']*100:.2f}%")
        if "lstm" in metrics:
            m = metrics["lstm"]
            with col_lstm:
                st.markdown("LSTM")
                st.write(f"MAPE: {m['mape']:.2f}%")
                st.write(f"MAE: {m['mae']:.5f}")
                st.write(f"MSE: {m['mse']:.6f}")
                if "acc" in m:
                    st.write(f"Accuracy (class): {m['acc']*100:.2f}%")
        if "hybrid" in metrics:
            m = metrics["hybrid"]
            w = metrics.get("weights", {})
            with col_hybrid:
                st.markdown("Hybrid (LGBM+LSTM)")
                st.write(f"MAPE: {m['mape']:.2f}%")
                st.write(f"MAE: {m['mae']:.5f}")
                st.write(f"MSE: {m['mse']:.6f}")
                if w:
                    st.write(f"Вес LGBM: {w.get('lgbm', 0.5):.2f}")
                    st.write(f"Вес LSTM: {w.get('lstm', 0.5):.2f}")
                if "acc" in m:
                    st.write(f"Accuracy (class): {m['acc']*100:.2f}%")
        if "svc" in metrics:
            m = metrics["svc"]
            with col_svc:
                st.markdown("SVC")
                if "acc" in m:
                    st.write(f"Accuracy (class): {m['acc']*100:.2f}%")
    st.plotly_chart(class_fig, use_container_width=True, key="classification_signals")
    if metrics:
        comparison_fig = build_classification_comparison_chart(
            df_model,
            model_data,
            model_weights,
            cls_conf_threshold,
        )
        st.plotly_chart(comparison_fig, use_container_width=True, key="classification_comparison")
    st.subheader("Лента новостей (Twitter и официальные источники)")
    if news_items:
        source_options = ["Twitter", "Federal Reserve", "ECB"]
        selected_sources = st.multiselect(
            "Источники",
            source_options,
            default=source_options,
        )
        if selected_sources:
            filtered_news = [
                n for n in news_items if n.get("publisher") in selected_sources
            ]
        else:
            filtered_news = news_items
        news_df = pd.DataFrame(
            [
                {
                    "Время": n.get("time"),
                    "Источник": n.get("publisher"),
                    "Заголовок": n.get("title"),
                    "Сентимент": n.get("score"),
                    "Важность": n.get("importance"),
                    "Эффект": n.get("effect"),
                }
                for n in filtered_news
            ]
        )
        st.dataframe(news_df, use_container_width=True)
    else:
        st.write("Новости из Twitter/официальных источников недоступны.")
    if show_lstm_chart and "lstm_reg_pred_test" in model_data and "lstm_y_reg_test" in model_data:
        lstm_fig = build_lstm_chart(df_model, model_data)
        st.plotly_chart(lstm_fig, use_container_width=True)
    signals = combine_signals(
        df_full,
        model_data,
        news_score,
        patterns,
        future_events_score,
        model_weights,
        price_model,
        classifier_override,
        cls_conf_threshold,
    )
    signals = enrich_signals_with_atr(signals, atr_info)
    if future_events:
        st.subheader("Будущие события (экономический календарь)")
        events_df = pd.DataFrame(
            [
                {
                    "Время": e["time"],
                    "Страна": e["country"],
                    "Событие": e["event"],
                    "Важность": e["impact"],
                    "Ожидание": e["estimate"],
                    "Предыдущее значение": e["previous"],
                    "Фактическое значение": e["actual"],
                }
                for e in future_events
            ]
        )
        st.dataframe(events_df, use_container_width=True)
    st.subheader("Обнаруженные фигуры и паттерны")
    for p in patterns:
        st.write("-", p)
    st.subheader("Итоговая рекомендация")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.metric("Текущая цена", f"{signals['last_price']:.5f}")
        st.metric("Цель на 7 дней", f"{signals['target_price']:.5f}")
        st.metric("Ожидаемая доходность", f"{signals['expected_return']*100:.2f}%")
    with col_right:
        st.write(f"Сигнал классификации: {signals['class_signal']} (-1 SELL, 0 HOLD, 1 BUY)")
        st.write(f"Вклад паттернов: {signals['pattern_bias']:+.2f}")
        st.write(f"Интегральный скоринг: {signals['score']:+.2f}")
        cls_used_internal = signals.get("cls_model_used")
        cls_used_label_map = {
            "svc": "SVC",
            "lstm": "LSTM",
            "lgbm": "LGBM",
            "hybrid": "Hybrid",
        }
        cls_used_label = cls_used_label_map.get(
            cls_used_internal, str(cls_used_internal).upper()
        )
        mode_text = (
            "Авто (лучшая по accuracy)"
            if cls_mode.startswith("Авто") and classifier_override is None
            else "Ручной выбор"
        )
        st.write(f"Модель классификации для сигнала: {cls_used_label}")
        st.write(f"Режим выбора модели: {mode_text}")
        if selected_profile is not None:
            st.write(f"Профиль таргета и классификации: {selected_profile}")
        else:
            st.write("Профиль таргета и классификации: не задан")
        st.markdown(f"### Рекомендация: **{signals['action']}**")
        st.write(f"Пояснение: {signals['reason']}")
    st.subheader("Информационная карточка сигнала")
    info_payload = {
        "ticker": ticker,
        "instrument_name": instrument_name,
        "profile": selected_profile,
        "years": years,
        "interval": interval,
        "horizon": horizon,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "news_score": float(news_score),
        "future_events_score": float(future_events_score),
        "patterns": patterns,
        "signal": {
            "action": signals["action"],
            "score": float(signals["score"]),
            "class_signal": int(signals["class_signal"]),
            "expected_return": float(signals["expected_return"]),
            "last_price": float(signals["last_price"]),
            "target_price": float(signals["target_price"]),
            "pattern_bias": int(signals["pattern_bias"]),
            "reason": signals["reason"],
            "cls_model_used": signals.get("cls_model_used"),
            "atr_14": float(signals["atr_14"]) if signals.get("atr_14") is not None else None,
            "atr_level": signals.get("atr_level"),
        },
    }
    col_info_left, col_info_right = st.columns([2, 1])
    with col_info_left:
        st.json(info_payload)
    with col_info_right:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Рекомендация": signals["action"],
                        "Сигнал (класс)": signals["class_signal"],
                        "Интегральный скоринг": signals["score"],
                        "Ожидаемая доходность": signals["expected_return"],
                        "Текущая цена": signals["last_price"],
                        "Целевая цена": signals["target_price"],
                        "ATR(14)": signals.get("atr_14"),
                        "Уровень волатильности": signals.get("atr_level"),
                        "Профиль": selected_profile,
                        "Модель классификации": cls_used_label,
                    }
                ]
            )
        )


api = FastAPI(title="FX AI Аналитика API")


@api.get("/predict")
def predict(
    ticker: str,
    instrument_name: str | None = None,
    years: int = 5,
    interval: str = "1d",
    profile: str | None = None,
):
    name = instrument_name or ticker
    try:
        result = get_signals_for_ticker(
            ticker=ticker,
            instrument_name=name,
            years=years,
            interval=interval,
            profile=profile,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@api.get("/history")
def get_history(ticker: str, years: int = 2, interval: str = "1d"):
    df = load_price_data(ticker, years=years, interval=interval)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    df = add_features(df)
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    dates = (
        df["Date"].dt.strftime("%Y-%m-%d").tolist()
        if "Date" in df.columns
        else df.index.astype(str).tolist()
    )
    open_series = df["Open"] if "Open" in df.columns else df["Close"]
    high_series = df["High"] if "High" in df.columns else df["Close"]
    low_series = df["Low"] if "Low" in df.columns else df["Close"]

    open_list = open_series.tolist()
    high_list = high_series.tolist()
    low_list = low_series.tolist()
    close_list = df["Close"].tolist()
    atr_list = df["ATR_14"].tolist() if "ATR_14" in df.columns else None
    ema_8_list = df["EMA_8"].tolist() if "EMA_8" in df.columns else None
    ema_21_list = df["EMA_21"].tolist() if "EMA_21" in df.columns else None
    ema_55_list = df["EMA_55"].tolist() if "EMA_55" in df.columns else None

    response = {
        "dates": dates,
        "open": open_list,
        "high": high_list,
        "low": low_list,
        "close": close_list,
        "atr": atr_list,
        "ema_8": ema_8_list,
        "ema_21": ema_21_list,
        "ema_55": ema_55_list,
    }

    def _clean_value(x):
        try:
            if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
                return None
        except Exception:
            return x
        return x

    for key, value in response.items():
        if isinstance(value, list):
            response[key] = [_clean_value(v) for v in value]

    return response


# Mount static files
api.mount("/static", StaticFiles(directory="static"), name="static")


@api.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")


def main():
    st.set_page_config(page_title="FX AI Аналитика", layout="wide")
    st.sidebar.title("FX AI Аналитика")
    page = st.sidebar.radio(
        "Страница",
        ["Major FX", "Crypto"],
        index=0,
    )
    if page == "Major FX":
        instruments = {
            "EURUSD": {"ticker": "EURUSD=X", "name": "EUR/USD"},
            "GBPUSD": {"ticker": "GBPUSD=X", "name": "GBP/USD"},
            "USDJPY": {"ticker": "USDJPY=X", "name": "USD/JPY"},
        }
    else:
        st.subheader("Криптовалюты")
        st.write("Скоро! Аналитика по Bitcoin находится в доработке.")
        return
    st.sidebar.subheader("Инструмент")
    selected_key = st.sidebar.selectbox(
        "Выберите инструмент",
        list(instruments.keys()),
        format_func=lambda k: instruments[k]["name"],
        key=f"instrument_{page}",
    )
    ticker = instruments[selected_key]["ticker"]
    instrument_name = instruments[selected_key]["name"]
    build_dashboard_for_ticker(ticker, instrument_name)


if __name__ == "__main__":
    main()
