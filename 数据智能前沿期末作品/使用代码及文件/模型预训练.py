import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.impute import SimpleImputer
import os

warnings.filterwarnings('ignore')



# ================== 数据预处理 ==================
def preprocess_data(df):
    """预处理NHANES数据集，提取特征并创建糖尿病标签"""
    print(f"原始数据形状: {df.shape}")

    # ================== 糖尿病诊断标准 ==================
    diabetes_criteria = (
            (df['LBXGH'].fillna(0) >= 6.5) |  # HbA1c
            (df['LBXGLT'].fillna(0) >= 126) |  # 空腹血糖
            (df['LBXSGL'].fillna(0) >= 200) |  # 随机血糖
            (df['DIQ010'].fillna(0) == 1) |  # 自我报告诊断
            (df['DIQ050'].fillna(0) == 1)  # 使用胰岛素（示例数据中没有，用0代替）
    )
    df['DIABETES'] = np.where(diabetes_criteria, 1, 0)

    # 创建代谢综合征特征
    df['METABOLIC_SYNDROME'] = 0
    if all(col in df.columns for col in ['BMXWAIST', 'RIAGENDR', 'BPXSY1', 'BPXDI1', 'LBXTR', 'LBDHDD', 'LBXGLT']):
        # 腹部肥胖 (男性>102cm, 女性>88cm)
        abdominal_obesity = np.where(df['RIAGENDR'] == 1, df['BMXWAIST'] > 102, df['BMXWAIST'] > 88)

        # 高血压 (SBP≥130或DBP≥85)
        hypertension = (df['BPXSY1'] >= 130) | (df['BPXDI1'] >= 85)

        # 高甘油三酯 (≥150 mg/dL)
        high_triglycerides = df['LBXTR'] >= 150

        # 低HDL (男性<40, 女性<50)
        if 'LBDHDD' in df.columns:
            low_hdl = np.where(df['RIAGENDR'] == 1, df['LBDHDD'] < 40, df['LBDHDD'] < 50)
        else:
            low_hdl = np.zeros(len(df), dtype=bool)

        # 空腹血糖升高 (≥100 mg/dL)
        high_fbg = df['LBXGLT'] >= 100

        # 满足3项即可诊断代谢综合征
        df['METABOLIC_SYNDROME'] = (abdominal_obesity.astype(int) +
                                    hypertension.astype(int) +
                                    high_triglycerides.astype(int) +
                                    low_hdl.astype(int) +
                                    high_fbg.astype(int)) >= 3

    # ================== 特征选择 ==================
    features = [
        'RIDAGEYR',  # 年龄
        'RIAGENDR',  # 性别
        'BMXBMI',  # BMI
        'PAQ650',  # 体力活动水平
        'SLQ060',  # 每月失眠天数
        'SMQ020',  # 吸烟状况
        'ALQ130',  # 酒精摄入
        'DIQ010',  # 糖尿病家族史
        'BMXWAIST',  # 腰围
        'HIGH_CHOL',  # 高胆固醇
        'HIGH_BP',  # 高血压
        'METABOLIC_SYNDROME'  # 代谢综合征
    ]

    # 创建高胆固醇特征
    if 'LBXTC' in df.columns:
        df['HIGH_CHOL'] = np.where(df['LBXTC'] >= 240, 1, 0)
    else:
        df['HIGH_CHOL'] = 0

    # 创建高血压特征
    if 'BPXSY1' in df.columns and 'BPXDI1' in df.columns:
        hypertension_criteria = (
                (df['BPXSY1'] >= 140) |
                (df['BPXDI1'] >= 90) |
                (df['BPQ040A'] == 1)
        )
        df['HIGH_BP'] = np.where(hypertension_criteria, 1, 0)
    else:
        df['HIGH_BP'] = 0

    # 检查特征是否存在
    available_features = []
    missing_features = []
    for feature in features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    if missing_features:
        print(f"警告: 以下特征在数据中不存在: {missing_features}")
        print("将使用可用的特征进行分析")

    # 处理缺失值
    df = df[available_features + ['DIABETES']].dropna(subset=['DIABETES'])

    # 数值型特征用中位数填充，分类变量用众数填充
    numerical_cols = [col for col in ['RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'PAQ650', 'ALQ130', 'SLQ060']
                      if col in df.columns]
    categorical_cols = [col for col in ['RIAGENDR', 'SMQ020', 'DIQ010', 'HIGH_CHOL', 'HIGH_BP', 'METABOLIC_SYNDROME']
                        if col in df.columns]

    if numerical_cols:
        df[numerical_cols] = SimpleImputer(strategy='median').fit_transform(df[numerical_cols])
    if categorical_cols:
        df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

    print(f"处理后数据形状: {df.shape}")
    print(f"糖尿病患者比例: {df['DIABETES'].mean():.2%}")

    # 转换分类变量
    df[categorical_cols] = df[categorical_cols].astype(int)

    # 使用统一的列名格式
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, prefix_sep='_', dtype=int)

    encoded_features = [col for col in df.columns if col != 'DIABETES']
    return df, encoded_features


# ================== 建模分析 ==================
def diabetes_analysis(df):
    X = df.drop('DIABETES', axis=1)
    y = df['DIABETES']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # 使用SMOTE过采样处理类别不平衡
    print("\n应用SMOTE过采样处理类别不平衡...")
    smote = SMOTE(random_state=42)

    # 使用梯度提升树
    model = GradientBoostingClassifier(
        random_state=42
    )

    # 创建管道
    pipeline = Pipeline([
        ('smote', smote),
        ('model', model)
    ])

    # 参数网格搜索
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__min_samples_split': [5, 10],
        'model__subsample': [0.7, 0.8]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    print("开始网格搜索优化参数...")
    grid_search.fit(X_train, y_train)

    # 最佳模型
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳F1分数: {grid_search.best_score_:.4f}")

    # 保存模型
    joblib.dump(best_model, 'diabetes_model_gb.pkl')

    # 评估最佳模型
    y_pred = best_model.predict(X_test)
    print("\n默认阈值模型评估报告：")
    print(classification_report(y_test, y_pred))

    if len(y_test.unique()) > 1:
        auc_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        print(f"AUC得分：{auc_score:.4f}")
    else:
        print("警告：测试集仅包含单一类别，无法计算AUC")
        auc_score = 0

    # 基于F1分数优化阈值
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 寻找最佳F1分数的阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # 使用调整后的阈值
    y_pred_adj = (y_proba > optimal_threshold).astype(int)

    print(f"\n调整阈值至{optimal_threshold:.2f}后的模型评估报告：")
    print(classification_report(y_test, y_pred_adj))

    # 计算F1分数
    f1 = f1_score(y_test, y_pred_adj)
    print(f"调整后F1分数: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred_adj)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非糖尿病', '糖尿病'],
                yticklabels=['非糖尿病', '糖尿病'])
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('糖尿病预测混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.grid(True)
    plt.savefig('pr_curve.png')
    plt.close()

    # 保存模型
    joblib.dump(best_model, 'model/diabetes_model_gb.pkl')

    # 保存特征列
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, 'model/feature_columns.pkl')

    # 保存模型元数据
    metadata = {
        'model_type': 'GradientBoostingClassifier',
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d"),
        'features': feature_columns,
        'best_params': grid_search.best_params_,
        'f1_score': f1,
        'auc_score': auc_score,
        'optimal_threshold': optimal_threshold
    }
    os.makedirs('model', exist_ok=True)
    joblib.dump(metadata, 'model/model_metadata.pkl')

    return best_model, optimal_threshold, f1, auc_score, feature_columns  # 返回特征列


# ================== 报告生成 ==================
def generate_report(patient_data, model, feature_names, threshold, f1_score, auc_score):
    patient_data = patient_data.reindex(columns=feature_names, fill_value=0)
    risk_score = model.predict_proba(patient_data)[0][1]

    # 关键指标提取
    base_info = {
        '年龄': patient_data['RIDAGEYR'].iloc[0] if 'RIDAGEYR' in patient_data.columns else 0,
        'BMI': patient_data['BMXBMI'].iloc[0] if 'BMXBMI' in patient_data.columns else 0,
        '腰围': patient_data['BMXWAIST'].iloc[0] if 'BMXWAIST' in patient_data.columns else 0,
        '体力活动': get_activity_level_text(patient_data),
        '失眠天数': patient_data['SLQ060'].iloc[0] if 'SLQ060' in patient_data.columns else 0,
        '吸烟状况': get_readable_smoking_status(patient_data),
        '家族史': get_readable_family_history(patient_data),
        '高胆固醇': '是' if patient_data.get('HIGH_CHOL_1', pd.Series([0])).iloc[0] == 1 else '否',
        '高血压': '是' if patient_data.get('HIGH_BP_1', pd.Series([0])).iloc[0] == 1 else '否',
        '代谢综合征': '是' if patient_data.get('METABOLIC_SYNDROME_1', pd.Series([0])).iloc[0] == 1 else '否'
    }

    # 风险等级评估
    if risk_score >= 0.7:
        risk_level = "高危"
        risk_desc = "您的糖尿病风险显著高于平均水平，建议立即咨询医生并进行全面检查"
    elif risk_score >= 0.4:
        risk_level = "中危"
        risk_desc = "您有中度糖尿病风险，建议改善生活方式并定期监测血糖"
    else:
        risk_level = "低危"
        risk_desc = "您的糖尿病风险较低，继续保持健康生活方式"

    # 主要风险因素分析
    risk_factors = []
    if base_info['BMI'] >= 25:
        risk_factors.append("超重/肥胖（BMI≥25）")
    if base_info['腰围'] > 90:
        risk_factors.append("腹部肥胖（腰围＞90cm）")
    if base_info['失眠天数'] > 5:
        risk_factors.append("频繁失眠（每月＞5天）")
    if base_info['高胆固醇'] == '是':
        risk_factors.append("高胆固醇")
    if base_info['高血压'] == '是':
        risk_factors.append("高血压")
    if base_info['代谢综合征'] == '是':
        risk_factors.append("代谢综合征")

    risk_factors_text = "\n".join([f"• {f}" for f in risk_factors]) if risk_factors else "• 未发现显著风险因素"

    # 个性化建议
    advice = []
    if base_info['BMI'] >= 25:
        advice.append("建议通过饮食控制+每周150分钟运动减重5-10%")
    if base_info['腰围'] > 90:
        advice.append("减少久坐，每天进行20分钟核心训练")
    if base_info['失眠天数'] > 5:
        advice.append("睡前1小时避免电子设备，保持卧室黑暗安静")
    if base_info['高胆固醇'] == '是':
        advice.append("减少饱和脂肪摄入，增加膳食纤维，考虑咨询医生是否需要药物治疗")
    if base_info['高血压'] == '是':
        advice.append("控制钠盐摄入，定期监测血压，遵医嘱服药")
    if base_info['代谢综合征'] == '是':
        advice.append("代谢综合征需要综合管理，包括饮食、运动和药物治疗")

    advice.append("每年至少检测1次空腹血糖和HbA1c")

    advice_text = "\n".join([f"• {a}" for a in advice])

    report = f"""
    ============ 糖尿病风险评估报告 ============

    您的糖尿病风险评分：{risk_score:.1%}
    风险等级：{risk_level}
    {risk_desc}

    关键健康指标：
    • 年龄: {base_info['年龄']}岁
    • BMI: {base_info['BMI']:.1f} ({'正常' if base_info['BMI'] < 25 else '超重' if base_info['BMI'] < 30 else '肥胖'})
    • 腰围: {base_info['腰围']}cm
    • 体力活动: {base_info['体力活动']}
    • 每月失眠天数: {base_info['失眠天数']}天
    • 吸烟状况: {base_info['吸烟状况']}
    • 糖尿病家族史: {base_info['家族史']}
    • 高胆固醇: {base_info['高胆固醇']}
    • 高血压: {base_info['高血压']}
    • 代谢综合征: {base_info['代谢综合征']}

    主要风险因素：
    {risk_factors_text}

    个性化建议：
    {advice_text}

    *本报告基于机器学习模型预测结果，不替代专业医疗建议
    *模型性能: F1分数={f1_score:.4f}, AUC={auc_score:.4f}, 阈值={threshold:.2f}
    """
    return report


# ================== 辅助函数 ==================
def get_activity_level_text(data):
    if 'PAQ650' in data.columns:
        level = data['PAQ650'].values[0]
        levels = {1: "几乎不活动", 2: "轻度活动", 3: "中度活动", 4: "重度活动"}
        return levels.get(level, f"未知级别({level})")
    return "未提供"


def get_readable_smoking_status(data):
    status_map = {
        'SMQ020_1': "每天吸烟",
        'SMQ020_2': "偶尔吸烟",
        'SMQ020_3': "从不吸烟",
        'SMQ020_4': "已戒烟"
    }
    for col in data.columns:
        if col.startswith('SMQ020_') and data[col].values[0] == 1:
            return status_map.get(col, "未知状态")
    return "未提供"


def get_readable_family_history(data):
    history_map = {
        'DIQ010_1': "有",
        'DIQ010_2': "无",
        'DIQ010_3': "不确定"
    }
    for col in data.columns:
        if col.startswith('DIQ010_') and data[col].values[0] == 1:
            return history_map.get(col, "未知状态")
    return "未提供"


# ================== 输入模块 ==================
def input_patient_data(available_features):
    print("\n请输入您的健康数据：")
    patient_data = {}

    # 基础特征输入
    if 'RIDAGEYR' in available_features:
        patient_data['RIDAGEYR'] = [get_valid_input("年龄（岁）：", int, 1, 120)]

    if 'RIAGENDR' in available_features:
        print("性别：\n1.男性\n2.女性")
        patient_data['RIAGENDR'] = [get_valid_input("选择：", int, choices=[1, 2])]

    if 'BMXBMI' in available_features:
        weight = get_valid_input("体重（kg）：", float, 20, 300)
        height = get_valid_input("身高（m）：", float, 0.5, 2.5)
        patient_data['BMXBMI'] = [round(weight / (height ** 2), 1)]
        print(f"BMI计算结果：{patient_data['BMXBMI'][0]:.1f}")

    if 'BMXWAIST' in available_features:
        patient_data['BMXWAIST'] = [get_valid_input("腰围（cm）：", float, 40, 200)]

    # 体力活动水平输入
    if 'PAQ650' in available_features:
        print("\n=== 体力活动水平 ===")
        print("1. 几乎不活动（如久坐工作，很少运动）")
        print("2. 轻度活动（如每周1-2次轻度运动，如散步）")
        print("3. 中度活动（如每周3-5次中等强度运动，如快走、骑自行车）")
        print("4. 重度活动（如每日高强度运动或体力劳动）")
        patient_data['PAQ650'] = [get_valid_input("请选择对应的数字（1-4）：", int, choices=[1, 2, 3, 4])]

    # 酒精摄入输入
    if 'ALQ130' in available_features:
        patient_data['ALQ130'] = [get_valid_input("过去12个月内，您平均每周饮酒多少杯？（1杯=14g纯酒精）：", float, 0, 100)]

    if 'SLQ060' in available_features:
        patient_data['SLQ060'] = [get_valid_input("过去一个月失眠天数：", int, 0, 30)]

    # 吸烟状况输入
    if any(col.startswith('SMQ020_') for col in available_features):
        print("\n=== 吸烟状况 ===")
        print("1. 每天吸烟")
        print("2. 偶尔吸烟")
        print("3. 从不吸烟")
        print("4. 已戒烟")
        choice = get_valid_input("请选择：", int, choices=[1, 2, 3, 4])
        for i in range(1, 5):
            col_name = f'SMQ020_{i}'
            if col_name in available_features:
                patient_data[col_name] = [1 if choice == i else 0]

    # 糖尿病家族史输入
    if any(col.startswith('DIQ010_') for col in available_features):
        print("\n=== 糖尿病家族史 ===")
        print("1. 有")
        print("2. 无")
        print("3. 不确定")
        choice = get_valid_input("请选择：", int, choices=[1, 2, 3])
        for i in range(1, 4):
            col_name = f'DIQ010_{i}'
            if col_name in available_features:
                patient_data[col_name] = [1 if choice == i else 0]

    # 高胆固醇输入
    if any(col.startswith('HIGH_CHOL_') for col in available_features):
        print("\n=== 高胆固醇 ===")
        print("您是否被诊断有高胆固醇（总胆固醇≥240 mg/dL）？")
        print("1. 是")
        print("2. 否")
        choice = get_valid_input("请选择：", int, choices=[1, 2])
        for i in range(1, 3):
            col_name = f'HIGH_CHOL_{i}'
            if col_name in available_features:
                patient_data[col_name] = [1 if choice == i else 0]

    # 高血压输入
    if any(col.startswith('HIGH_BP_') for col in available_features):
        print("\n=== 高血压 ===")
        print("您是否被诊断有高血压（收缩压≥140 mmHg或舒张压≥90 mmHg）？")
        print("1. 是")
        print("2. 否")
        choice = get_valid_input("请选择：", int, choices=[1, 2])
        for i in range(1, 3):
            col_name = f'HIGH_BP_{i}'
            if col_name in available_features:
                patient_data[col_name] = [1 if choice == i else 0]

    # 代谢综合征输入
    if any(col.startswith('METABOLIC_SYNDROME_') for col in available_features):
        print("\n=== 代谢综合征 ===")
        print("您是否有以下至少3项？")
        print("1. 腹部肥胖（男性腰围>102cm, 女性>88cm）")
        print("2. 高血压（≥130/85 mmHg）")
        print("3. 高甘油三酯（≥150 mg/dL）")
        print("4. 低HDL胆固醇（男性<40, 女性<50 mg/dL）")
        print("5. 空腹血糖升高（≥100 mg/dL）")
        print("1. 是")
        print("2. 否")
        choice = get_valid_input("请选择：", int, choices=[1, 2])
        for i in range(1, 3):
            col_name = f'METABOLIC_SYNDROME_{i}'
            if col_name in available_features:
                patient_data[col_name] = [1 if choice == i else 0]

    # 转换为DataFrame
    patient_df = pd.DataFrame(patient_data)
    full_data = pd.DataFrame(0, index=[0], columns=available_features)

    # 处理数值型特征
    numerical_cols = [c for c in available_features if not any(c.startswith(prefix) for prefix in
                                                               ['RIAGENDR_', 'SMQ020_', 'DIQ010_', 'HIGH_CHOL_',
                                                                'HIGH_BP_', 'METABOLIC_SYNDROME_'])]
    for col in numerical_cols:
        if col in patient_df.columns:
            full_data[col] = patient_df[col]

    # 处理分类特征
    for col in patient_df.columns:
        if col in available_features:
            full_data[col] = patient_df[col]

    return full_data


def get_valid_input(prompt, data_type, min_val=None, max_val=None, choices=None):
    while True:
        try:
            value = data_type(input(prompt))
            if min_val is not None and value < min_val:
                print(f"值不能小于{min_val}")
            elif max_val is not None and value > max_val:
                print(f"值不能大于{max_val}")
            elif choices and value not in choices:
                print(f"请输入{choices}中的选项")
            else:
                return value
        except ValueError:
            print(f"请输入有效的{data_type.__name__}类型值")


# ================== 主程序 ==================
def main():
    try:
        data_files = ["demographic.csv", "diet.csv", "examination.csv", "labs.csv", "questionnaire.csv"]
        if all(os.path.exists(f) for f in data_files):
            print("正在读取现有数据文件...")
            dfs = {
                "demo": pd.read_csv("demographic.csv"),
                "diet": pd.read_csv("diet.csv"),
                "exam": pd.read_csv("examination.csv"),
                "labs": pd.read_csv("labs.csv"),
                "ques": pd.read_csv("questionnaire.csv")
            }
            alldata = reduce(lambda left, right: pd.merge(left, right, on="SEQN", how="outer"), dfs.values())


        processed_df, encoded_features = preprocess_data(alldata)

        model, optimal_threshold, f1_score, auc_score, feature_columns = diabetes_analysis(processed_df)

        # 用户交互生成报告
        if input("\n生成个性化报告？(y/n): ").lower() == 'y':
            # 使用保存的特征列
            user_data = input_patient_data(feature_columns)
            user_report = generate_report(user_data, model, feature_columns, optimal_threshold, f1_score, auc_score)
            print(f"\n{user_report}")
            with open('personal_report.txt', 'w', encoding='utf-8') as f:
                f.write(user_report)
            print("报告已保存至 personal_report.txt")


            user_data.to_csv('user_data.csv', index=False)
            print("用户数据已保存至 user_data.csv")

    except Exception as e:
        print(f"程序错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()