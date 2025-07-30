from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
import io

app = Flask(__name__)


model = None
feature_columns = None
threshold = 0.3
f1_score = 0.0
auc_score = 0.0


def initialize_model():
    global model, feature_columns, threshold, f1_score, auc_score

    model_path = 'model/diabetes_model_gb.pkl'
    feature_path = 'model/feature_columns.pkl'

    if os.path.exists(model_path) and os.path.exists(feature_path):
        print("加载预训练模型...")
        model = joblib.load(model_path)
        feature_columns = joblib.load(feature_path)

        if os.path.exists('model/model_metadata.pkl'):
            metadata = joblib.load('model/model_metadata.pkl')
            threshold = metadata.get('threshold', 0.3)
            f1_score = metadata.get('f1_score', 0.0)
            auc_score = metadata.get('auc_score', 0.0)
    else:
        print("未找到预训练模型，开始训练新模型...")
        model, feature_columns, threshold, f1_score, auc_score = train_dummy_model()

        os.makedirs('model', exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(feature_columns, feature_path)
        joblib.dump({
            'threshold': threshold,
            'f1_score': f1_score,
            'auc_score': auc_score
        }, 'model/model_metadata.pkl')


def train_dummy_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import f1_score as f1_score_func  # 重命名避免冲突
    from sklearn.metrics import roc_auc_score as roc_auc_score_func  # 重命名避免冲突
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

    feature_columns = [
        'RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'PAQ650', 'SLQ060',
        'SMQ020_1', 'SMQ020_2', 'SMQ020_3', 'SMQ020_4',
        'DIQ010_1', 'DIQ010_2', 'DIQ010_3',
        'HIGH_CHOL_1', 'HIGH_CHOL_2',
        'HIGH_BP_1', 'HIGH_BP_2',
        'METABOLIC_SYNDROME_1', 'METABOLIC_SYNDROME_2'
    ]

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    f1 = f1_score_func(y, y_pred)
    auc = roc_auc_score_func(y, model.predict_proba(X)[:, 1])

    return model, feature_columns, 0.35, f1, auc


initialize_model()


# 辅助函数
def get_activity_level_text(level):
    levels = {
        1: "几乎不活动",
        2: "轻度活动",
        3: "中度活动",
        4: "重度活动"
    }
    return levels.get(level, f"未知级别({level})")


def get_smoking_status(status):
    status_map = {
        1: "每天吸烟",
        2: "偶尔吸烟",
        3: "从不吸烟",
        4: "已戒烟"
    }
    return status_map.get(status, "未知状态")


def get_family_history(history):
    history_map = {
        1: "有",
        2: "无",
        3: "不确定"
    }
    return history_map.get(history, "未知状态")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_report', methods=['POST'])
def generate_report():
    form_data = request.form
    raw_data = {
        'age': int(form_data.get('age', 40)),
        'gender': int(form_data.get('gender', 1)),
        'bmi': float(form_data.get('bmi', 25.0)),
        'waist': float(form_data.get('waist', 85.0)),
        'activity': int(form_data.get('activity', 2)),
        'insomnia': int(form_data.get('insomnia', 3)),
        'alcohol': float(form_data.get('alcohol', 2.0)),
        'smoking': int(form_data.get('smoking', 3)),
        'family_history': int(form_data.get('family_history', 2)),
        'high_chol': int(form_data.get('high_chol', 2)),
        'high_bp': int(form_data.get('high_bp', 2)),
        'metabolic_syndrome': int(form_data.get('metabolic_syndrome', 2))
    }

    # 创建用于预测的DataFrame
    user_data = {
        'RIDAGEYR': [raw_data['age']],
        'RIAGENDR': [raw_data['gender']],
        'BMXBMI': [raw_data['bmi']],
        'BMXWAIST': [raw_data['waist']],
        'PAQ650': [raw_data['activity']],
        'SLQ060': [raw_data['insomnia']],
        'ALQ130': [raw_data['alcohol']],
        'SMQ020': [raw_data['smoking']],
        'DIQ010': [raw_data['family_history']],
        'HIGH_CHOL': [raw_data['high_chol']],
        'HIGH_BP': [raw_data['high_bp']],
        'METABOLIC_SYNDROME': [raw_data['metabolic_syndrome']]
    }

    df = pd.DataFrame(user_data)
    for col in ['SMQ020', 'DIQ010', 'HIGH_CHOL', 'HIGH_BP', 'METABOLIC_SYNDROME']:
        value = df[col].iloc[0]
        df[f'{col}_{value}'] = 1

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    risk_score = model.predict_proba(df)[0][1]

    report_data = {
        'risk_score': risk_score,
        'age': raw_data['age'],
        'bmi': raw_data['bmi'],
        'waist': raw_data['waist'],
        'activity': get_activity_level_text(raw_data['activity']),
        'insomnia': raw_data['insomnia'],
        'smoking': get_smoking_status(raw_data['smoking']),
        'family_history': get_family_history(raw_data['family_history']),
        'high_chol': "是" if raw_data['high_chol'] == 1 else "否",
        'high_bp': "是" if raw_data['high_bp'] == 1 else "否",
        'metabolic_syndrome': "是" if raw_data['metabolic_syndrome'] == 1 else "否",
        'f1_score': f1_score,
        'auc_score': auc_score,
        'threshold': threshold
    }

    # 风险等级评估
    if risk_score >= 0.7:
        report_data['risk_level'] = "高危"
        report_data['risk_desc'] = "您的糖尿病风险显著高于平均水平，建议立即咨询医生并进行全面检查"
    elif risk_score >= 0.4:
        report_data['risk_level'] = "中危"
        report_data['risk_desc'] = "您有中度糖尿病风险，建议改善生活方式并定期监测血糖"
    else:
        report_data['risk_level'] = "低危"
        report_data['risk_desc'] = "您的糖尿病风险较低，继续保持健康生活方式"

    # 主要风险因素分析
    risk_factors = []
    if report_data['bmi'] >= 25:
        risk_factors.append("超重/肥胖（BMI≥25）")
    if report_data['waist'] > 90:
        risk_factors.append("腹部肥胖（腰围＞90cm）")
    if report_data['insomnia'] > 5:
        risk_factors.append("频繁失眠（每月＞5天）")
    if report_data['high_chol'] == '是':
        risk_factors.append("高胆固醇")
    if report_data['high_bp'] == '是':
        risk_factors.append("高血压")
    if report_data['metabolic_syndrome'] == '是':
        risk_factors.append("代谢综合征")

    report_data['risk_factors'] = "\n".join([f"• {f}" for f in risk_factors]) if risk_factors else "• 未发现显著风险因素"

    # 个性化建议
    advice = []
    if report_data['bmi'] >= 25:
        advice.append("建议通过饮食控制+每周150分钟运动减重5-10%")
    if report_data['waist'] > 90:
        advice.append("减少久坐，每天进行20分钟核心训练")
    if report_data['insomnia'] > 5:
        advice.append("睡前1小时避免电子设备，保持卧室黑暗安静")
    if report_data['high_chol'] == '是':
        advice.append("减少饱和脂肪摄入，增加膳食纤维，考虑咨询医生是否需要药物治疗")
    if report_data['high_bp'] == '是':
        advice.append("控制钠盐摄入，定期监测血压，遵医嘱服药")
    if report_data['metabolic_syndrome'] == '是':
        advice.append("代谢综合征需要综合管理，包括饮食、运动和药物治疗")

    advice.append("每年至少检测1次空腹血糖和HbA1c")

    report_data['advice'] = "\n".join([f"• {a}" for a in advice])

    # BMI分类
    if report_data['bmi'] < 18.5:
        bmi_class = "体重不足"
    elif report_data['bmi'] < 25:
        bmi_class = "正常"
    elif report_data['bmi'] < 30:
        bmi_class = "超重"
    else:
        bmi_class = "肥胖"
    report_data['bmi_class'] = bmi_class

    # 创建文本报告
    text_report = f"""
    ============ 糖尿病风险评估报告 ============

    您的糖尿病风险评分：{report_data['risk_score']:.1%}
    风险等级：{report_data['risk_level']}
    {report_data['risk_desc']}

    关键健康指标：
    • 年龄: {report_data['age']}岁
    • BMI: {report_data['bmi']:.1f} ({report_data['bmi_class']})
    • 腰围: {report_data['waist']}cm
    • 体力活动: {report_data['activity']}
    • 每月失眠天数: {report_data['insomnia']}天
    • 吸烟状况: {report_data['smoking']}
    • 糖尿病家族史: {report_data['family_history']}
    • 高胆固醇: {report_data['high_chol']}
    • 高血压: {report_data['high_bp']}
    • 代谢综合征: {report_data['metabolic_syndrome']}

    主要风险因素：
    {report_data['risk_factors']}

    个性化建议：
    {report_data['advice']}

    *本报告基于机器学习模型预测结果，不替代专业医疗建议
    *模型性能: F1分数={report_data['f1_score']:.4f}, AUC={report_data['auc_score']:.4f}, 阈值={report_data['threshold']:.2f}
    """

    report_data['text_report'] = text_report
    return render_template('report.html', report=report_data)


@app.route('/download_report')
def download_report():
    report_text = request.args.get('text', '')

    # 创建内存文件
    report_io = io.BytesIO()
    report_io.write(report_text.encode('utf-8'))
    report_io.seek(0)

    return send_file(
        report_io,
        as_attachment=True,
        download_name='糖尿病风险评估报告.txt',
        mimetype='text/plain'
    )


if __name__ == '__main__':
    app.run(debug=True)