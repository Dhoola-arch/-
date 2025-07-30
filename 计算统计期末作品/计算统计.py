import numpy as np
import pandas as pd
import os
import gc
import warnings
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from hmmlearn import hmm
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from prophet import Prophet
import holidays

# 配置matplotlib使用Agg后端（无GUI）
matplotlib.use('Agg')

# 忽略非关键警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置参数
SAVE_PLOTS = True
CITIES = ['Delhi', 'Chennai', 'Bengaluru', 'Ahmedabad', 'Gurugram', 'Hyderabad', 'Patna', 'Visakhapatnam']
NORTH_CITIES = ['Delhi', 'Gurugram', 'Lucknow', 'Patna']  # 北方农业城市
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
TRAIN_START = '2015-01-01'
TRAIN_END = '2018-12-31'
TEST_START = '2019-01-01'
TEST_END = '2019-12-31'

# 配置 matplotlib 字体
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.figsize': (12, 6),
    'figure.dpi': 100
})


# 更准确的AQI估算函数
def calculate_aqi(row):
    """基于主要污染物计算AQI的简化方法"""
    main_pollutant = row[POLLUTANTS].idxmax()
    main_value = row[main_pollutant]

    factors = {
        'PM2.5': 1.2, 'PM10': 1.2,
        'NO2': 1.5, 'SO2': 1.5,
        'CO': 10, 'O3': 1.3,
        'NO': 1.4, 'NOx': 1.4
    }
    return main_value * factors.get(main_pollutant, 1.3)


def aqi_to_category(aqi):
    """将AQI值转换为类别"""
    if pd.isna(aqi):
        return 'Unknown'
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'


def enhanced_preprocessing(city, city_df):
    """增强型预处理 - 特别处理北方农业城市"""
    print(f"Applying enhanced preprocessing for {city}...")

    # 确保日期列已转换为datetime
    if not pd.api.types.is_datetime64_any_dtype(city_df['Date']):
        city_df['Date'] = pd.to_datetime(city_df['Date'])

    # 保存原始索引
    original_index = city_df.index

    # 设置日期索引用于时间插值
    city_df.set_index('Date', inplace=True)

    # 创建时间特征（确保在预处理中可用）
    city_df['Year'] = city_df.index.year
    city_df['Month'] = city_df.index.month

    # 北方农业城市特殊处理
    if city in NORTH_CITIES:
        # 创建秸秆焚烧季标志 (10-11月)
        city_df['Harvest_Season'] = city_df['Month'].isin([10, 11]).astype(int)

        # 特殊插值处理
        for pollutant in POLLUTANTS:
            # 使用时间序列插值方法
            city_df[pollutant] = city_df[pollutant].interpolate(method='time', limit_direction='both')

            # 秸秆焚烧季使用邻近值插值
            harvest_mask = (city_df['Harvest_Season'] == 1) & city_df[pollutant].isna()
            city_df.loc[harvest_mask, pollutant] = city_df[pollutant].ffill().bfill()

    # 多重插补法处理缺失值
    imputer = IterativeImputer(max_iter=15, random_state=42, skip_complete=True)
    pollutant_data = city_df[POLLUTANTS].values
    imputed_data = imputer.fit_transform(pollutant_data)
    city_df[POLLUTANTS] = imputed_data

    # 处理极端值 - 基于城市特定阈值
    if 'AQI' in city_df.columns:
        aqi_q1 = city_df['AQI'].quantile(0.05)
        aqi_q3 = city_df['AQI'].quantile(0.95)
        city_df['AQI'] = city_df['AQI'].clip(lower=aqi_q1, upper=aqi_q3)

    # 恢复原始索引
    city_df.reset_index(inplace=True)

    return city_df


def prepare_city_data(city, df):
    """为特定城市准备数据"""
    print(f"Preparing data for {city}...")

    # 筛选城市数据
    city_df = df[df['City'] == city].copy().sort_values('Date').reset_index(drop=True)

    # 应用增强型预处理
    city_df = enhanced_preprocessing(city, city_df)

    # 填充AQI缺失值
    if 'AQI' in city_df.columns:
        city_df['AQI'] = city_df.apply(
            lambda x: x['AQI'] if not pd.isna(x['AQI']) else calculate_aqi(x), axis=1
        )
    else:
        # 如果AQI列不存在，则创建它
        city_df['AQI'] = city_df.apply(calculate_aqi, axis=1)

    # 填充AQI_Bucket缺失值
    if 'AQI_Bucket' in city_df.columns:
        city_df['AQI_Bucket'] = city_df['AQI_Bucket'].fillna(
            city_df['AQI'].apply(aqi_to_category)
        )
    else:
        city_df['AQI_Bucket'] = city_df['AQI'].apply(aqi_to_category)

    # 确保日期列已转换
    if not pd.api.types.is_datetime64_any_dtype(city_df['Date']):
        city_df['Date'] = pd.to_datetime(city_df['Date'])

    # 设置日期索引用于创建时间特征
    city_df.set_index('Date', inplace=True)

    # 创建时间特征
    city_df['Year'] = city_df.index.year
    city_df['Month'] = city_df.index.month
    city_df['Weekday'] = city_df.index.weekday
    city_df['DayOfYear'] = city_df.index.dayofyear
    city_df['IsWeekend'] = (city_df['Weekday'] >= 5).astype(int)
    city_df['Season'] = city_df['Month'].apply(lambda m: (m % 12 + 3) // 3)

    # 添加上下文特征
    city_df['PrevDay_AQI'] = city_df['AQI'].shift(1)
    if city_df['PrevDay_AQI'].isna().any():
        city_df['PrevDay_AQI'].fillna(method='bfill', inplace=True)

    # 印度特定假期特征
    try:
        india_holidays = holidays.India(years=city_df['Year'].unique())
        city_df['Is_Holiday'] = city_df.index.to_series().apply(lambda d: d in india_holidays).astype(int)
    except Exception as e:
        print(f"Holiday processing failed: {str(e)}")
        city_df['Is_Holiday'] = 0

    # 北方城市添加秸秆焚烧季特征
    if city in NORTH_CITIES:
        city_df['Paddy_Burning'] = (city_df['Month'].isin([10, 11])).astype(int)

    # 恢复原始索引
    city_df.reset_index(inplace=True)

    # 确保没有缺失值
    city_df.fillna(method='ffill', inplace=True)
    city_df.fillna(method='bfill', inplace=True)

    return city_df

def split_train_test(city_df):
    """按时间划分训练集和测试集"""
    # 确保日期列已转换
    if not pd.api.types.is_datetime64_any_dtype(city_df['Date']):
        city_df['Date'] = pd.to_datetime(city_df['Date'])

    # 划分训练集（前四年）
    train_mask = (city_df['Date'] >= TRAIN_START) & (city_df['Date'] <= TRAIN_END)
    train_df = city_df[train_mask].copy()

    # 划分测试集（最后一年）
    test_mask = (city_df['Date'] >= TEST_START) & (city_df['Date'] <= TEST_END)
    test_df = city_df[test_mask].copy()

    if len(train_df) > 0:
        print(
            f"Training set: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} days)")
    else:
        print("Warning: No training data found")

    if len(test_df) > 0:
        print(f"Test set: {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} days)")
    else:
        print("Warning: No test data found")

    return train_df, test_df


def get_pollution_profile(train_df):
    """分析城市污染特征以选择最佳模型"""
    profile = {}

    try:
        # 计算稳定性 (标准差)
        profile['stability'] = 1 / train_df['AQI'].std()

        # 计算峰值数量 (超过平均值+标准差)
        aqi_mean = train_df['AQI'].mean()
        aqi_std = train_df['AQI'].std()
        profile['peak_count'] = (train_df['AQI'] > aqi_mean + aqi_std).sum()

        # 计算季节性强度
        # 确保日期索引已设置
        aqi_series = train_df.set_index('Date')['AQI']
        stl = STL(aqi_series, period=365, seasonal=13, robust=True)
        res = stl.fit()
        var_resid = np.nanvar(res.resid)
        var_total = np.nanvar(aqi_series)
        profile['seasonal_strength'] = max(0, 1 - var_resid / var_total)

        # 计算自相关性
        profile['autocorrelation'] = train_df['AQI'].autocorr(lag=7)

        print(f"Pollution profile - Stability: {profile['stability']:.4f}, "
              f"Peaks: {profile['peak_count']}, "
              f"Seasonal: {profile['seasonal_strength']:.4f}, "
              f"Autocorr: {profile['autocorrelation']:.4f}")
    except Exception as e:
        print(f"Error creating pollution profile: {str(e)}")
        # 默认值
        profile = {
            'stability': 0.5,
            'peak_count': 10,
            'seasonal_strength': 0.3,
            'autocorrelation': 0.6
        }

    return profile


def train_models(train_df, pollution_profile):
    """根据污染特征训练最合适的模型"""
    print("Training models based on pollution profile...")

    # 准备特征和目标
    features = POLLUTANTS + ['Year', 'Month', 'DayOfYear', 'Weekday', 'IsWeekend', 'Season', 'PrevDay_AQI']

    # 北方城市添加特殊特征
    if 'Paddy_Burning' in train_df.columns:
        features.append('Paddy_Burning')

    # 确保所有特征都存在
    available_features = [f for f in features if f in train_df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Training set missing features: {missing}")

    X_train = train_df[available_features]
    y_train = train_df['AQI'].values

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled)

    # 根据污染特征选择模型
    model_type = "Ensemble"  # 默认

    try:
        # 规则1: 高稳定性 + 强季节性 -> Prophet
        if pollution_profile['stability'] > 0.5 and pollution_profile['seasonal_strength'] > 0.4:
            model_type = "Prophet"
            print("Selecting Prophet model for stable and seasonal data")

        # 规则2: 多峰值 -> 集成模型
        elif pollution_profile['peak_count'] > 20:
            model_type = "Ensemble"
            print("Selecting Ensemble model for peaky data")

        # 规则3: 低自相关性 -> 复杂模型 (此处简化为集成)
        elif pollution_profile['autocorrelation'] < 0.4:
            model_type = "Ensemble"
            print("Selecting Ensemble model for low-autocorrelation data")
    except Exception as e:
        print(f"Error in model selection: {str(e)}")
        model_type = "Ensemble"

    # 训练选择的模型
    if model_type == "Prophet":
        # Prophet模型将在单独函数中训练
        return None, None, None, scaler, available_features, model_type

    else:  # 默认训练集成模型
        # 1. 训练LASSO模型
        lasso = LassoCV(
            cv=5, alphas=np.logspace(-4, 0, 50),
            max_iter=10000, tol=1e-3, random_state=42, selection='random'
        )
        lasso.fit(X_train_scaled, y_train)

        # 2. 训练随机森林模型
        rf = RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)

        # 模型集成函数
        def ensemble_predict(X):
            return (lasso.predict(X) * 0.4 + rf.predict(X) * 0.6)

        return lasso, rf, ensemble_predict, scaler, available_features, model_type


def evaluate_models(models, test_df, model_type):
    """在测试集上评估模型性能"""
    print(f"Evaluating {model_type} model on test set...")

    metrics = {}
    ensemble_pred = None

    # Prophet模型单独处理
    if model_type == "Prophet":
        # Prophet模型在forecast_with_prophet中已评估
        return metrics, ensemble_pred

    # 确保models有足够的值解包
    if models is None or len(models) < 5:
        print(f"Error: Not enough models to unpack ({len(models) if models else 0} < 5)")
        return metrics, ensemble_pred

    try:
        lasso, rf, ensemble_predict, scaler, features = models[:5]

        # 准备测试数据
        available_features = [f for f in features if f in test_df.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            print(f"Warning: Test set missing features: {missing}")

        X_test = test_df[available_features]
        y_test = test_df['AQI'].values

        # 标准化
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled)

        # 预测
        lasso_pred = lasso.predict(X_test_scaled)
        rf_pred = rf.predict(X_test_scaled)
        ensemble_pred = ensemble_predict(X_test_scaled)

        # 计算指标
        for name, pred in zip(['LASSO', 'Random Forest', 'Ensemble'],
                              [lasso_pred, rf_pred, ensemble_pred]):
            metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                'MAE': mean_absolute_error(y_test, pred),
                'R2': r2_score(y_test, pred)
            }
            print(
                f"{name}: RMSE={metrics[name]['RMSE']:.2f}, MAE={metrics[name]['MAE']:.2f}, R2={metrics[name]['R2']:.4f}")
    except Exception as e:
        print(f"Error evaluating models: {str(e)}")

    return metrics, ensemble_pred


def train_hmm(train_df):
    """训练隐马尔可夫模型（仅使用训练集）"""
    print("Training Hidden Markov Model...")

    # 创建状态映射
    aqi_categories = {
        'Good': 0, 'Satisfactory': 1, 'Moderate': 2,
        'Poor': 3, 'Very Poor': 4, 'Severe': 5
    }
    train_df['State'] = train_df['AQI_Bucket'].map(aqi_categories)

    # 使用多个污染物作为观测值
    obs_features = POLLUTANTS + ['AQI']

    # 确保所有观测特征都存在
    available_obs_features = [f for f in obs_features if f in train_df.columns]
    if len(available_obs_features) < len(obs_features):
        missing = set(obs_features) - set(available_obs_features)
        print(f"Warning: HMM training set missing observation features: {missing}")

    observations = train_df[available_obs_features].values

    # 标准化观测值
    scaler = StandardScaler()
    observations_scaled = scaler.fit_transform(observations)

    # 训练多变量HMM
    model = hmm.GaussianHMM(
        n_components=6, covariance_type="diag", n_iter=500
    )
    model.fit(observations_scaled)

    return model, scaler, available_obs_features


def apply_hmm(model, scaler, df, obs_features):
    """应用HMM模型到整个数据集"""
    # 确保所有观测特征都存在
    available_obs_features = [f for f in obs_features if f in df.columns]
    if len(available_obs_features) < len(obs_features):
        missing = set(obs_features) - set(available_obs_features)
        print(f"Warning: HMM application missing observation features: {missing}")
        for feature in missing:
            df[feature] = 0.0
        available_obs_features = obs_features

    observations = df[available_obs_features].values

    # 标准化观测值
    observations_scaled = scaler.transform(observations)

    # 解码状态序列
    hidden_states = model.predict(observations_scaled)
    df['Predicted_State'] = hidden_states.astype(int)

    return df


def forecast_with_prophet(train_df, test_df):
    """使用Prophet进行时间序列预测"""
    print("Forecasting with Prophet...")

    # 准备数据
    prophet_train = pd.DataFrame({
        'ds': train_df['Date'],
        'y': train_df['AQI']
    })

    # 创建并拟合模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,  # 提高转折点灵敏度
        seasonality_mode='multiplicative'
    )

    try:
        model.add_country_holidays(country_name='IN')
    except Exception:
        print("Failed to add holidays, proceeding without")

    try:
        model.fit(prophet_train)
    except Exception as e:
        print(f"Prophet model fitting failed: {str(e)}")
        return None, None, None, None, None

    # 创建未来数据框
    future_dates = pd.date_range(start=TEST_START, end=TEST_END)
    future = pd.DataFrame({'ds': future_dates})

    # 进行预测
    try:
        forecast = model.predict(future)
    except Exception as e:
        print(f"Prophet forecasting failed: {str(e)}")
        return None, None, None, None, None

    # 提取预测结果
    forecast_dates = forecast['ds']
    forecast_mean = forecast['yhat'].values
    forecast_lower = forecast['yhat_lower'].values
    forecast_upper = forecast['yhat_upper'].values

    # 计算评估指标
    try:
        # 合并测试数据
        test_dates = test_df['Date'].values
        test_aqi = test_df['AQI'].values

        # 对齐日期
        aligned_indices = []
        for date in forecast_dates:
            idx = np.where(test_dates == date)[0]
            if len(idx) > 0:
                aligned_indices.append(idx[0])

        actual = test_aqi[aligned_indices]
        predicted = forecast_mean[:len(aligned_indices)]

        valid_mask = ~np.isnan(actual) & ~np.isnan(predicted)

        if np.sum(valid_mask) > 0:
            rmse = np.sqrt(mean_squared_error(actual[valid_mask], predicted[valid_mask]))
            mae = mean_absolute_error(actual[valid_mask], predicted[valid_mask])
            r2 = r2_score(actual[valid_mask], predicted[valid_mask])
        else:
            rmse = mae = r2 = np.nan

        print(f"Prophet model performance:")
        print(f"RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        return forecast_dates, forecast_mean, forecast_lower, forecast_upper, {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    except Exception as e:
        print(f"Error evaluating Prophet: {str(e)}")
        return forecast_dates, forecast_mean, forecast_lower, forecast_upper, None


def stl_seasonal_analysis(train_df, city):
    """使用STL进行时间序列季节性分析"""
    print("Performing STL seasonal decomposition...")

    # 确保数据按日期排序
    try:
        train_series = train_df.set_index('Date')['AQI'].sort_index()
    except Exception as e:
        print(f"Error setting index for STL: {str(e)}")
        return None, 0

    # 进行STL分解
    try:
        stl = STL(train_series, period=365, seasonal=13, robust=True)
        res = stl.fit()

        # 计算季节性强度
        var_resid = np.nanvar(res.resid)
        var_total = np.nanvar(train_series)
        seasonal_strength = max(0, 1 - var_resid / var_total)
        print(f"Seasonal strength: {seasonal_strength:.4f}")

        # 可视化分解结果
        fig = res.plot()
        fig.set_size_inches(14, 10)
        fig.suptitle(f'{city} AQI STL Decomposition', fontsize=16)
        plt.tight_layout()

        # 确保目录存在
        os.makedirs(f"results/{city}", exist_ok=True)
        plt.savefig(f"results/{city}/stl_decomposition.png")
        plt.close(fig)

        return res, seasonal_strength
    except Exception as e:
        print(f"STL decomposition failed: {str(e)}")
        return None, 0


def save_visualizations(city, train_df, test_df, models, hmm_model,
                        forecast_results, obs_features, stl_decomposition,
                        model_type, seasonal_strength):
    """保存可视化结果"""
    print("Saving visualizations...")
    os.makedirs(f"results/{city}", exist_ok=True)

    # 特征重要性可视化（仅适用于集成模型）
    if model_type != "Prophet" and models and models[0] is not None and len(models) >= 5:
        try:
            lasso, rf, _, scaler, features = models[:5]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # LASSO特征重要性
            ax1.bar(features, lasso.coef_)
            ax1.set_title(f'{city} - LASSO Feature Importance')
            ax1.tick_params(axis='x', rotation=45)
            ax1.set_ylabel('Coefficient Value')
            ax1.grid(True, linestyle='--', alpha=0.3)

            # 随机森林特征重要性
            rf_importances = rf.feature_importances_
            ax2.bar(features, rf_importances)
            ax2.set_title(f'{city} - Random Forest Feature Importance')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylabel('Importance Score')
            ax2.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'results/{city}/feature_importance_comparison.png')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving feature importance: {str(e)}")

    # 应用HMM模型到整个数据集
    full_df = pd.concat([train_df, test_df])
    try:
        if hmm_model and len(hmm_model) == 3:
            full_df = apply_hmm(hmm_model[0], hmm_model[1], full_df, obs_features)
    except Exception as e:
        print(f"Error applying HMM: {str(e)}")

    # AQI时间序列与HMM状态
    try:
        fig, ax = plt.subplots(figsize=(14, 7))

        # 绘制训练集和测试集
        ax.plot(train_df['Date'], train_df['AQI'], label='Training Data', color='blue', alpha=0.7)
        ax.plot(test_df['Date'], test_df['AQI'], label='Test Data', color='green', alpha=0.7)

        # 确保状态值不为零
        if 'Predicted_State' in full_df.columns:
            state_values = full_df['Predicted_State'] * (
                        full_df['AQI'].max() / max(full_df['Predicted_State'].max(), 1))
            scatter = ax.scatter(full_df['Date'], state_values,
                                 c=full_df['Predicted_State'], cmap='viridis',
                                 label='HMM States', alpha=0.6)

        # 添加训练/测试分割线
        if len(test_df) > 0:
            split_date = test_df['Date'].min()
            ax.axvline(x=split_date, color='red', linestyle='--', label='Train/Test Split')

        ax.set_title(f'{city} AQI Time Series with HMM States')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI Value')
        ax.grid(True, linestyle='--', alpha=0.3)
        if 'Predicted_State' in full_df.columns:
            plt.colorbar(scatter, ax=ax, label='HMM State')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'results/{city}/aqi_hmm_states.png')
        plt.close(fig)
    except Exception as e:
        print(f"Error saving HMM visualization: {str(e)}")

    # Prophet预测结果可视化
    if forecast_results and forecast_results[0] is not None:
        forecast_dates, forecast_mean, forecast_lower, forecast_upper, metrics = forecast_results

        try:
            fig, ax = plt.subplots(figsize=(14, 7))

            # 绘制训练集
            ax.plot(train_df['Date'], train_df['AQI'], label='Training Data', color='blue')

            # 绘制测试集实际值
            if len(test_df) > 0:
                ax.plot(test_df['Date'], test_df['AQI'], label='Actual AQI', color='green')

            # 绘制预测值
            ax.plot(forecast_dates, forecast_mean, label='Prophet Forecast', color='red')

            # 置信区间
            ax.fill_between(forecast_dates, forecast_lower, forecast_upper, color='red', alpha=0.1)

            # 添加训练/测试分割线
            if len(test_df) > 0:
                split_date = test_df['Date'].min()
                ax.axvline(x=split_date, color='purple', linestyle='--', label='Train/Test Split')

            ax.set_title(f'{city} Prophet Forecast vs Actual')
            ax.set_xlabel('Date')
            ax.set_ylabel('AQI Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'results/{city}/prophet_forecast_vs_actual.png')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving Prophet visualization: {str(e)}")

    # 季节性强度可视化
    if seasonal_strength > 0:
        try:
            plt.figure(figsize=(8, 5))
            plt.bar(['Seasonal Strength'], [seasonal_strength], color='skyblue')
            plt.ylim(0, 1)
            plt.title(f'{city} Seasonal Strength')
            plt.ylabel('Strength Score')
            plt.grid(True, axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'results/{city}/seasonal_strength.png')
            plt.close()
        except Exception as e:
            print(f"Error saving seasonal strength: {str(e)}")

    print(f"Visualizations saved for: {city}")


def analyze_city(city, df):
    """分析单个城市的数据"""
    print(f"\n{'=' * 50}")
    print(f"Starting analysis: {city}")
    print(f"{'=' * 50}")

    try:
        # 准备数据
        city_df = prepare_city_data(city, df)

        # 按时间划分训练集和测试集
        train_df, test_df = split_train_test(city_df)

        # 如果测试集为空，跳过该城市
        if len(test_df) == 0:
            print(f"Warning: {city} has no data between {TEST_START} and {TEST_END}, skipping")
            return {
                'city': city,
                'error': f"No test data between {TEST_START} and {TEST_END}"
            }

        # 分析污染特征
        pollution_profile = get_pollution_profile(train_df)

        # 根据污染特征训练模型
        models = train_models(train_df, pollution_profile)
        model_type = models[5] if models and len(models) > 5 else "Ensemble"

        # 在测试集上评估模型
        if model_type == "Prophet":
            # Prophet模型训练和评估
            forecast_results = forecast_with_prophet(train_df, test_df)
            metrics = {'Prophet': forecast_results[4]} if forecast_results and forecast_results[4] else None
            ensemble_pred = None
        else:
            metrics, ensemble_pred = evaluate_models(models, test_df, model_type)
            forecast_results = forecast_with_prophet(train_df, test_df)

        # 训练HMM模型
        try:
            hmm_model, hmm_scaler, obs_features = train_hmm(train_df)
        except Exception as e:
            print(f"HMM training failed: {str(e)}")
            hmm_model, hmm_scaler, obs_features = None, None, None

        # 进行季节性分解分析
        stl_decomposition, seasonal_strength = stl_seasonal_analysis(train_df, city)

        # 保存可视化结果
        save_visualizations(city, train_df, test_df, models,
                            (hmm_model, hmm_scaler, obs_features) if hmm_model else None,
                            forecast_results, obs_features, stl_decomposition,
                            model_type, seasonal_strength)

        # 返回结果摘要
        return {
            'city': city,
            'train_start': train_df['Date'].min().date() if not train_df.empty else None,
            'train_end': train_df['Date'].max().date() if not train_df.empty else None,
            'test_start': test_df['Date'].min().date() if not test_df.empty else None,
            'test_end': test_df['Date'].max().date() if not test_df.empty else None,
            'model_type': model_type,
            'metrics': metrics,
            'forecast_metrics': forecast_results[4] if forecast_results and forecast_results[4] else None,
            'seasonal_strength': seasonal_strength,
            'pollution_profile': pollution_profile
        }

    except Exception as e:
        print(f"Error analyzing {city}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'city': city,
            'error': str(e)
        }
    finally:
        # 清理内存
        gc.collect()


def main():
    """主函数"""
    print("Starting air quality analysis...")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing period: {TEST_START} to {TEST_END}")

    # 加载数据
    print("Loading data...")
    try:
        df = pd.read_csv('city_day.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Data loaded successfully. Total records: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # 分析每个城市
    results = []
    for city in CITIES:
        result = analyze_city(city, df)
        if result:
            results.append(result)

    # 打印摘要报告
    print("\nAnalysis Summary Report:")
    print("-" * 120)
    print(
        f"{'City':<15}{'Model':<10}{'Train Days':<12}{'Test Days':<12}{'Ensemble RMSE':<15}{'Prophet RMSE':<15}{'Seasonal Strength':<18}{'Status'}")
    print("-" * 120)

    for res in results:
        if 'error' in res:
            print(f"{res['city']:<15}{'Error':<10}{'':<12}{'':<12}{'':<15}{'':<15}{'':<18}{res['error'][:30]}")
        else:
            try:
                train_days = (res['train_end'] - res['train_start']).days + 1 if res['train_start'] and res[
                    'train_end'] else 0
            except:
                train_days = 0

            try:
                test_days = (res['test_end'] - res['test_start']).days + 1 if res['test_start'] and res[
                    'test_end'] else 0
            except:
                test_days = 0

            # 获取指标
            ensemble_rmse = np.nan
            if res['metrics'] and 'Ensemble' in res['metrics']:
                ensemble_rmse = res['metrics']['Ensemble']['RMSE']

            prophet_rmse = np.nan
            if res['forecast_metrics']:
                prophet_rmse = res['forecast_metrics']['RMSE']

            seasonal_strength = res['seasonal_strength'] if 'seasonal_strength' in res else np.nan

            print(f"{res['city']:<15}{res['model_type']:<10}{train_days:<12}{test_days:<12}"
                  f"{ensemble_rmse:<15.2f}{prophet_rmse:<15.2f}"
                  f"{seasonal_strength:<18.4f}{'Success'}")

    print("\nAnalysis completed!")


if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    main()