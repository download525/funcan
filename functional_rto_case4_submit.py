import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("train_2.csv")
SUBMIT_PATH = Path("test.csv")
REPORT_PATH = Path("functional_rto_case4_report.json")

ID_COL = "new_id"
YEAR_COL = "Год"
MONTH_COL = "Месяц"
TARGET_COL = "РТО"

WINDOW = 12
RIDGE_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
EPS = 1.0


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(100 * np.mean(np.abs((y_pred - y_true) / np.maximum(y_true, EPS))))


def trapz_weights(t):
    w = np.empty_like(t)
    w[1:-1] = (t[2:] - t[:-2]) / 2
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    return w


def basis_matrix(length):
    t = np.linspace(0, 1, length)
    cols = [
        np.ones(length),
        t - t.mean(),
        (t - t.mean()) ** 2,
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        np.sin(4 * np.pi * t),
        np.cos(4 * np.pi * t),
        np.sin(6 * np.pi * t),
        np.cos(6 * np.pi * t),
    ]
    return np.column_stack(cols), trapz_weights(t)


BASIS, WEIGHTS = basis_matrix(WINDOW)


def functional_projection(values):
    values = np.asarray(values, dtype=float)
    return values @ (BASIS * WEIGHTS[:, None])


def interval_means(values, bins=4):
    values = np.asarray(values, dtype=float)
    parts = np.array_split(values, bins)
    return np.array([p.mean() for p in parts], dtype=float)


def series_features(values, prefix):
    values = np.asarray(values, dtype=float)
    proj = functional_projection(values)
    bins = interval_means(values, 4)
    diffs = np.diff(values)
    out = []
    names = []

    for i, v in enumerate(proj):
        out.append(v)
        names.append(f"{prefix}_func_{i}")
    for i, v in enumerate(bins):
        out.append(v)
        names.append(f"{prefix}_interval_{i}")

    stats = {
        "last": values[-1],
        "prev": values[-2],
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "trend": values[-1] - values[0],
        "diff_last": diffs[-1],
        "diff_mean": diffs.mean(),
        "last_minus_mean": values[-1] - values.mean(),
    }
    for key, val in stats.items():
        out.append(val)
        names.append(f"{prefix}_{key}")
    return np.array(out, dtype=float), names


def numeric_frame(df):
    numeric_cols = [
        c
        for c in df.columns
        if c not in {ID_COL, YEAR_COL, MONTH_COL, TARGET_COL}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def encode_categories(last_rows, cols):
    mats = []
    names = []
    for col in cols:
        counts = last_rows[col].astype(str).value_counts()
        keep = counts.index[:30]
        vals = last_rows[col].astype(str)
        for cat in keep:
            mats.append((vals == cat).astype(float).to_numpy())
            names.append(f"{col}={cat}")
    if not mats:
        return np.empty((len(last_rows), 0)), []
    return np.column_stack(mats), names


def prepare_store_arrays(df):
    df = df.sort_values([ID_COL, YEAR_COL, MONTH_COL]).reset_index(drop=True)
    months = df[[YEAR_COL, MONTH_COL]].drop_duplicates().sort_values([YEAR_COL, MONTH_COL])
    month_pairs = list(map(tuple, months[[YEAR_COL, MONTH_COL]].to_numpy()))
    date_to_pos = {pair: i for i, pair in enumerate(month_pairs)}
    n_months = len(month_pairs)

    ids = np.sort(df[ID_COL].unique())
    id_to_row = {store_id: i for i, store_id in enumerate(ids)}
    rto = np.full((len(ids), n_months), np.nan, dtype=float)

    for row in df[[ID_COL, YEAR_COL, MONTH_COL, TARGET_COL]].itertuples(index=False):
        i = id_to_row[getattr(row, ID_COL)]
        j = date_to_pos[(getattr(row, YEAR_COL), getattr(row, MONTH_COL))]
        rto[i, j] = getattr(row, TARGET_COL)

    last_rows = (
        df.sort_values([ID_COL, YEAR_COL, MONTH_COL])
        .groupby(ID_COL, as_index=False)
        .tail(1)
        .sort_values(ID_COL)
        .reset_index(drop=True)
    )
    return df, ids, month_pairs, rto, last_rows


def build_region_ratio(df, current_pair, target_pair, last_rows):
    cur = df[(df[YEAR_COL] == current_pair[0]) & (df[MONTH_COL] == current_pair[1])][
        [ID_COL, TARGET_COL, "Регион", "Населенный пункт"]
    ].rename(columns={TARGET_COL: "cur"})
    tgt = df[(df[YEAR_COL] == target_pair[0]) & (df[MONTH_COL] == target_pair[1])][
        [ID_COL, TARGET_COL]
    ].rename(columns={TARGET_COL: "tgt"})
    ratio_df = cur.merge(tgt, on=ID_COL, how="inner")
    ratio_df["ratio"] = ratio_df["tgt"] / ratio_df["cur"].replace(0, np.nan)
    global_ratio = float(ratio_df["ratio"].median())
    region_ratio = ratio_df.groupby("Регион")["ratio"].median()
    city_ratio = ratio_df.groupby("Населенный пункт")["ratio"].median()

    rr = last_rows["Регион"].map(region_ratio).fillna(global_ratio).to_numpy(dtype=float)
    cr = last_rows["Населенный пункт"].map(city_ratio).fillna(global_ratio).to_numpy(dtype=float)
    return global_ratio, rr, cr


def make_samples(df, ids, month_pairs, rto, last_rows, hs, return_meta=False):
    log_rto = np.log1p(rto)
    numeric_cols = numeric_frame(df)
    last_numeric = last_rows[numeric_cols].fillna(last_rows[numeric_cols].median(numeric_only=True))
    last_numeric = last_numeric.to_numpy(dtype=float)

    cat_cols = [
        c
        for c in ["Дата открытия, категориальный", "Торговая площадь, категориальный", "Регион"]
        if c in last_rows.columns
    ]
    cat_mat, cat_names = encode_categories(last_rows, cat_cols)

    X_all = []
    y_all = []
    cur_all = []
    seasonal_all = []
    meta = []
    names = None

    for h in hs:
        target_h = h + 1
        next_year, next_month = month_pairs[target_h]
        cur_pair_prev_year = (month_pairs[h][0] - 1, month_pairs[h][1])
        tgt_pair_prev_year = (next_year - 1, next_month)

        try:
            prev_cur_idx = month_pairs.index(cur_pair_prev_year)
            prev_tgt_idx = month_pairs.index(tgt_pair_prev_year)
            store_ratio = rto[:, prev_tgt_idx] / np.maximum(rto[:, prev_cur_idx], EPS)
        except ValueError:
            store_ratio = np.full(len(ids), np.nan)

        current = rto[:, h]
        target = rto[:, target_h] if target_h < rto.shape[1] else np.full(len(ids), np.nan)
        valid = np.isfinite(current) & np.isfinite(target)

        if np.isfinite(store_ratio).any():
            ratio_med = np.nanmedian(store_ratio)
        else:
            ratio_med = 1.0
        store_ratio = np.nan_to_num(store_ratio, nan=ratio_med, posinf=ratio_med, neginf=ratio_med)
        store_ratio = np.clip(store_ratio, 0.85, 1.35)

        if cur_pair_prev_year in month_pairs and tgt_pair_prev_year in month_pairs:
            global_ratio, region_ratio, city_ratio = build_region_ratio(df, cur_pair_prev_year, tgt_pair_prev_year, last_rows)
        else:
            global_ratio = ratio_med
            region_ratio = np.full(len(ids), ratio_med)
            city_ratio = np.full(len(ids), ratio_med)

        month_angle = 2 * np.pi * next_month / 12
        month_feats = np.tile(
            np.array([np.sin(month_angle), np.cos(month_angle), next_month / 12], dtype=float),
            (len(ids), 1),
        )

        rows = []
        for i in range(len(ids)):
            window = log_rto[i, h - WINDOW + 1 : h + 1]
            f_rto, f_names = series_features(window, "log_rto")
            ratio_feats = np.array(
                [
                    store_ratio[i],
                    region_ratio[i],
                    city_ratio[i],
                    global_ratio,
                    np.log(store_ratio[i]),
                    np.log(region_ratio[i]),
                    np.log(city_ratio[i]),
                    np.log(global_ratio),
                    np.log1p(current[i]),
                ],
                dtype=float,
            )
            row = np.concatenate([f_rto, ratio_feats, month_feats[i], last_numeric[i], cat_mat[i]])
            rows.append(row)
            if names is None:
                names = (
                    f_names
                    + [
                        "store_prev_year_ratio",
                        "region_prev_year_ratio",
                        "city_prev_year_ratio",
                        "global_prev_year_ratio",
                        "log_store_prev_year_ratio",
                        "log_region_prev_year_ratio",
                        "log_city_prev_year_ratio",
                        "log_global_prev_year_ratio",
                        "log_current_rto",
                    ]
                    + ["next_month_sin", "next_month_cos", "next_month_scaled"]
                    + numeric_cols
                    + cat_names
                )

        X_h = np.vstack(rows)
        X_all.append(X_h[valid])
        y_all.append(np.log1p(target[valid]) - np.log1p(current[valid]))
        cur_all.append(current[valid])
        seasonal_all.append((current * store_ratio)[valid])
        if return_meta:
            meta.extend([(ids[i], h, target_h) for i in np.where(valid)[0]])

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    cur = np.concatenate(cur_all)
    seasonal = np.concatenate(seasonal_all)
    if return_meta:
        return X, y, cur, seasonal, names, meta
    return X, y, cur, seasonal, names


def make_prediction_frame(df, ids, month_pairs, rto, last_rows):
    log_rto = np.log1p(rto)
    numeric_cols = numeric_frame(df)
    last_numeric = last_rows[numeric_cols].fillna(last_rows[numeric_cols].median(numeric_only=True))
    last_numeric = last_numeric.to_numpy(dtype=float)
    cat_cols = [
        c
        for c in ["Дата открытия, категориальный", "Торговая площадь, категориальный", "Регион"]
        if c in last_rows.columns
    ]
    cat_mat, cat_names = encode_categories(last_rows, cat_cols)

    h = len(month_pairs) - 1
    next_month = 3
    cur_pair_prev_year = (2024, 2)
    tgt_pair_prev_year = (2024, 3)
    prev_cur_idx = month_pairs.index(cur_pair_prev_year)
    prev_tgt_idx = month_pairs.index(tgt_pair_prev_year)
    store_ratio = rto[:, prev_tgt_idx] / np.maximum(rto[:, prev_cur_idx], EPS)
    ratio_med = np.nanmedian(store_ratio)
    store_ratio = np.nan_to_num(store_ratio, nan=ratio_med, posinf=ratio_med, neginf=ratio_med)
    store_ratio = np.clip(store_ratio, 0.85, 1.35)
    global_ratio, region_ratio, city_ratio = build_region_ratio(df, cur_pair_prev_year, tgt_pair_prev_year, last_rows)

    current = rto[:, h]
    month_angle = 2 * np.pi * next_month / 12
    month_feats = np.tile(np.array([np.sin(month_angle), np.cos(month_angle), next_month / 12], dtype=float), (len(ids), 1))

    rows = []
    for i in range(len(ids)):
        window = log_rto[i, h - WINDOW + 1 : h + 1]
        f_rto, _ = series_features(window, "log_rto")
        ratio_feats = np.array(
            [
                store_ratio[i],
                region_ratio[i],
                city_ratio[i],
                global_ratio,
                np.log(store_ratio[i]),
                np.log(region_ratio[i]),
                np.log(city_ratio[i]),
                np.log(global_ratio),
                np.log1p(current[i]),
            ],
            dtype=float,
        )
        rows.append(np.concatenate([f_rto, ratio_feats, month_feats[i], last_numeric[i], cat_mat[i]]))
    return np.vstack(rows), current, current * store_ratio, current * region_ratio, current * city_ratio


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    return mu, sigma


def ridge_train(X, y, alpha):
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(X1.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(X1.T @ X1 + alpha * penalty, X1.T @ y)


def ridge_predict(X, beta):
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


def fit_model(X, y, alpha):
    mu, sigma = standardize_fit(X)
    Xs = (X - mu) / sigma
    beta = ridge_train(Xs, y, alpha)
    return {"mu": mu, "sigma": sigma, "beta": beta, "alpha": alpha}


def apply_model(model, X):
    Xs = (X - model["mu"]) / model["sigma"]
    return ridge_predict(Xs, model["beta"])


def main():
    print("read data")
    df = pd.read_csv(DATA_PATH)
    df, ids, month_pairs, rto, last_rows = prepare_store_arrays(df)
    print("stores:", len(ids), "months:", len(month_pairs), month_pairs[0], month_pairs[-1])

    train_hs = list(range(WINDOW, len(month_pairs) - 2))
    val_hs = [len(month_pairs) - 2]

    X_train, y_train, cur_train, seasonal_train, names = make_samples(df, ids, month_pairs, rto, last_rows, train_hs)
    X_val, y_val, cur_val, seasonal_val, _ = make_samples(df, ids, month_pairs, rto, last_rows, val_hs)
    y_val_abs = cur_val * np.expm1(y_val)
    y_val_abs = np.expm1(np.log1p(cur_val) + y_val)

    results = []
    best = None
    for alpha in RIDGE_GRID:
        model = fit_model(X_train, y_train, alpha)
        ratio_pred = apply_model(model, X_val)
        pred = np.expm1(np.log1p(cur_val) + ratio_pred)
        pred = np.maximum(pred, 0)
        score = mape(y_val_abs, pred)
        results.append({"alpha": alpha, "val_mape": score})
        print("alpha", alpha, "val_mape", score)
        if best is None or score < best["score"]:
            best = {"alpha": alpha, "score": score}

    best_alpha = best["alpha"]
    val_model = fit_model(X_train, y_train, best_alpha)
    val_model_pred = np.maximum(np.expm1(np.log1p(cur_val) + apply_model(val_model, X_val)), 0)
    seasonal_val = np.maximum(seasonal_val, 0)

    blend_scores = []
    for w in np.linspace(0, 1, 41):
        pred = w * val_model_pred + (1 - w) * seasonal_val
        blend_scores.append((mape(y_val_abs, pred), float(w)))
    best_blend_score, best_blend_weight = min(blend_scores)
    print("best alpha:", best_alpha)
    print("model val mape:", mape(y_val_abs, val_model_pred))
    print("seasonal val mape:", mape(y_val_abs, seasonal_val))
    print("blend weight:", best_blend_weight, "blend val mape:", best_blend_score)

    y_train_abs = np.expm1(np.log1p(cur_train) + y_train)
    residual_target = np.log1p(y_train_abs) - np.log1p(np.maximum(seasonal_train, EPS))
    residual_results = []
    residual_best = None
    for alpha in RIDGE_GRID:
        res_model = fit_model(X_train, residual_target, alpha)
        res_pred = apply_model(res_model, X_val)
        pred = np.maximum(np.expm1(np.log1p(np.maximum(seasonal_val, EPS)) + res_pred), 0)
        score = mape(y_val_abs, pred)
        residual_results.append({"alpha": alpha, "val_mape": score})
        print("residual alpha", alpha, "val_mape", score)
        if residual_best is None or score < residual_best["score"]:
            residual_best = {"alpha": alpha, "score": score}

    full_hs = list(range(WINDOW, len(month_pairs) - 1))
    X_full, y_full, cur_full, seasonal_full, _ = make_samples(df, ids, month_pairs, rto, last_rows, full_hs)
    final_model = fit_model(X_full, y_full, best_alpha)
    y_full_abs = np.expm1(np.log1p(cur_full) + y_full)
    residual_full = np.log1p(y_full_abs) - np.log1p(np.maximum(seasonal_full, EPS))
    final_residual_model = fit_model(X_full, residual_full, residual_best["alpha"])

    X_pred, current, store_seasonal, region_seasonal, city_seasonal = make_prediction_frame(df, ids, month_pairs, rto, last_rows)
    ratio_final = apply_model(final_model, X_pred)
    model_pred = np.maximum(np.expm1(np.log1p(current) + ratio_final), 0)
    seasonal_pred = np.maximum(store_seasonal, 0)
    residual_final = apply_model(final_residual_model, X_pred)
    residual_pred = np.maximum(np.expm1(np.log1p(np.maximum(seasonal_pred, EPS)) + residual_final), 0)

    pd.DataFrame({ID_COL: ids.astype(int), "rto": model_pred}).to_csv(
        "test_functional_ratio.csv", index=False
    )
    pd.DataFrame({ID_COL: ids.astype(int), "rto": residual_pred}).to_csv(
        "test_functional_residual.csv", index=False
    )

    candidates = {
        "seasonal": seasonal_pred,
        "functional_ratio": model_pred,
        "functional_residual": residual_pred,
    }
    if residual_best["score"] < best_blend_score:
        final_name = "functional_residual"
        final_pred = residual_pred
    else:
        final_name = "seasonal"
        final_pred = seasonal_pred
    final_pred = np.maximum(final_pred, 0)

    submit = pd.DataFrame({ID_COL: ids.astype(int), "rto": final_pred})
    submit.to_csv(SUBMIT_PATH, index=False)

    report = {
        "rows": int(len(submit)),
        "months": [list(map(int, p)) for p in month_pairs],
        "window": WINDOW,
        "feature_count": int(X_full.shape[1]),
        "train_examples": int(X_full.shape[0]),
        "validation_target": "2025-02",
        "ridge_grid": results,
        "best_alpha": best_alpha,
        "model_val_mape": mape(y_val_abs, val_model_pred),
        "seasonal_val_mape": mape(y_val_abs, seasonal_val),
        "blend_weight_model": best_blend_weight,
        "blend_val_mape": best_blend_score,
        "residual_ridge_grid": residual_results,
        "best_residual_alpha": residual_best["alpha"],
        "residual_val_mape": residual_best["score"],
        "chosen_final_method": final_name,
        "submit_path": str(SUBMIT_PATH),
        "functional_ratio_submit_path": "test_functional_ratio.csv",
        "functional_residual_submit_path": "test_functional_residual.csv",
        "prediction_min": float(submit["rto"].min()),
        "prediction_mean": float(submit["rto"].mean()),
        "prediction_max": float(submit["rto"].max()),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
