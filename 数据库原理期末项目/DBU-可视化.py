import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import os
from datetime import datetime
import time
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from textwrap import wrap

# 设置全局样式
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']  # 更好的中文支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['figure.dpi'] = 200  # 提高图像质量
mpl.rcParams['savefig.dpi'] = 300  # 保存图像的分辨率
mpl.rcParams['font.size'] = 6  # 全局字体大小

# 创建自定义字体属性
small_font = FontProperties(size=8)
medium_font = FontProperties(size=10)
large_font = FontProperties(size=12)
title_font = FontProperties(size=14, weight='bold')

# API配置
BASE_URL = "http://localhost:9999"  # 根据您的实际API地址修改
API_ENDPOINTS = {
    "device_usage_frequency": "/analysis/device-usage-frequency/",
    "concurrent_devices": "/analysis/concurrent-devices/",
    "area_usage_impact": "/analysis/area-usage-impact/",
    "security_event_stats": "/analysis/security-event-stats/",
    "device_usage_pattern": "/analysis/device-usage-pattern/",
    "energy_consumption": "/analysis/energy-consumption/",
    "concurrent_device_usage": "/analysis/concurrent-device-usage/",
    "user_activity": "/analysis/user-activity/",
    "device_failure_rate": "/analysis/device-failure-rate/",
    "feedback_sentiment": "/analysis/feedback-sentiment/",
}


def get_api_data(endpoint_key):
    """从API获取数据"""
    try:
        url = BASE_URL + API_ENDPOINTS[endpoint_key]
        response = requests.get(url, timeout=300)
        response.raise_for_status()  # 检查HTTP错误
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取 {endpoint_key} 数据失败: {str(e)}")
        return None
    except ValueError:
        print(f"解析 {endpoint_key} 的JSON响应失败")
        return None


def create_output_directory():
    """创建输出目录"""
    output_dir = "smart_home_visualizations"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def wrap_labels(labels, max_width=15):
    """自动换行长标签"""
    return ['\n'.join(wrap(label, max_width)) for label in labels]


def visualize_device_usage_frequency(data, save_path=None):
    """可视化设备使用频率数据"""
    if not data or len(data) == 0:
        print("没有可用的设备使用频率数据")
        return

    df = pd.DataFrame(data)

    # 如果设备太多，只显示前20个
    if len(df['device_name'].unique()) > 20:
        top_devices = df.groupby('device_name')['frequency'].sum().nlargest(20).index
        df = df[df['device_name'].isin(top_devices)]
        print(f"设备数量过多，只显示前20个设备")

    pivot_df = df.pivot_table(index='device_name', columns='time_slot', values='frequency', fill_value=0)

    # 对设备名称进行换行处理
    wrapped_index = wrap_labels(pivot_df.index, 12)
    pivot_df.index = wrapped_index

    fig, ax = plt.subplots(figsize=(16, 10))

    pivot_df.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('设备使用频率按时间段分布', fontproperties=title_font)
    ax.set_ylabel('使用次数', fontproperties=medium_font)
    ax.set_xlabel('设备名称', fontproperties=medium_font)
    ax.tick_params(axis='x', labelsize=8, rotation=45)

    # 优化图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='时间段', bbox_to_anchor=(1.05, 1), loc='upper left',
              prop=small_font, title_fontproperties=medium_font)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'device_usage_frequency.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_concurrent_devices(data, save_path=None):
    """可视化同时使用的设备组合"""
    if not data or len(data) == 0:
        print("没有可用的同时使用设备数据")
        return

    # 如果组合太多，只显示前15个
    if len(data) > 15:
        data = sorted(data, key=lambda x: x['concurrent_count'], reverse=True)[:15]
        print(f"组合数量过多，只显示前15个组合")

    df = pd.DataFrame(data)
    df['device_pair'] = df['device1_name'] + ' & ' + df['device2_name']

    # 对设备组合名称进行换行处理
    df['device_pair'] = wrap_labels(df['device_pair'], 20)

    plt.figure(figsize=(12, 8))
    plt.barh(df['device_pair'], df['concurrent_count'], color='skyblue')
    plt.title('经常同时使用的设备组合', fontproperties=title_font)
    plt.xlabel('同时使用次数', fontproperties=medium_font)
    plt.ylabel('设备组合', fontproperties=medium_font)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'concurrent_devices.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 创建网络图
    plt.figure(figsize=(14, 10))
    G = nx.Graph()

    for item in data:
        G.add_edge(item['device1_name'], item['device2_name'], weight=item['concurrent_count'])

    if len(G.nodes) == 0:
        print("没有足够的节点创建网络图")
        return

    pos = nx.spring_layout(G, seed=42, k=0.3)  # 增加k值使节点更分散

    # 节点大小基于度数
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_size = [d * 5000 / max_degree for d in degrees.values()]

    # 边宽度基于权重
    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_width = [w * 10 / max_weight for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='gray', alpha=0.7)

    # 节点标签换行处理
    labels = {node: '\n'.join(wrap(node, 12)) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family='sans-serif')

    # 添加边权重标签
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title('设备同时使用关系网络图', fontproperties=title_font)
    plt.axis('off')

    if save_path:
        plt.savefig(os.path.join(save_path, 'concurrent_devices_network.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_area_usage_impact(data, save_path=None):
    """可视化房屋面积对设备使用的影响"""
    if not data or len(data) == 0:
        print("没有可用的房屋面积影响数据")
        return

    df = pd.DataFrame(data)

    # 如果设备类型太多，只显示前10个
    if len(df['device_type'].unique()) > 10:
        top_devices = df.groupby('device_type')['usage_count'].sum().nlargest(10).index
        df = df[df['device_type'].isin(top_devices)]
        print(f"设备类型过多，只显示前10个设备类型")

    # 平均使用时长
    plt.figure(figsize=(16, 10))
    for device_type in df['device_type'].unique():
        subset = df[df['device_type'] == device_type]
        plt.plot(subset['area_range'], subset['avg_duration'], marker='o', markersize=8,
                 label='\n'.join(wrap(device_type, 15)))

    plt.title('不同房屋面积下各类设备的平均使用时长', fontproperties=title_font)
    plt.ylabel('平均使用时长 (分钟)', fontproperties=medium_font)
    plt.xlabel('房屋面积范围 (平方米)', fontproperties=medium_font)
    plt.legend(title='设备类型', bbox_to_anchor=(1.05, 1), loc='upper left',
               prop=small_font, title_fontproperties=medium_font)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'area_avg_duration.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 使用次数热力图 - 修复了fmt参数问题
    plt.figure(figsize=(14, 10))
    pivot_df = df.pivot_table(index='device_type', columns='area_range', values='usage_count', fill_value=0)

    # 对设备类型进行换行处理
    wrapped_index = wrap_labels(pivot_df.index, 15)
    pivot_df.index = wrapped_index

    # 检查数据类型并动态设置fmt
    if pivot_df.values.dtype.kind in 'iu':  # 整数类型
        fmt = 'd'
    else:
        fmt = '.0f'  # 浮点数但显示为整数

    sns.heatmap(pivot_df, annot=True, fmt=fmt, cmap="YlGnBu", linewidths=.5, annot_kws={"size": 8})
    plt.title('不同面积房屋中各类设备的使用次数', fontproperties=title_font)
    plt.xlabel('房屋面积范围', fontproperties=medium_font)
    plt.ylabel('设备类型', fontproperties=medium_font)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'area_usage_count.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_security_event_stats(data, save_path=None):
    """可视化安防事件统计数据"""
    if not data or len(data) == 0:
        print("没有可用的安防事件数据")
        return

    df = pd.DataFrame(data)

    # 如果事件类型太多，合并小事件
    if len(df) > 10:
        threshold = df['count'].sum() * 0.03  # 3%阈值
        small_events = df[df['count'] < threshold]
        if not small_events.empty:
            other_count = small_events['count'].sum()
            other_percentage = small_events['percentage'].sum()
            df = df[df['count'] >= threshold]
            df = pd.concat([df, pd.DataFrame([{
                'event_type': '其他事件',
                'count': other_count,
                'percentage': other_percentage
            }])])
            print(f"合并了 {len(small_events)} 个小事件为'其他事件'")

    # 饼图
    plt.figure(figsize=(12, 10))
    plt.pie(df['count'], labels=df['event_type'], autopct='%1.1f%%',
            startangle=90, shadow=True, explode=[0.05] * len(df),
            colors=sns.color_palette('pastel', len(df)), textprops={'fontsize': 8})
    plt.title('安防事件类型分布', fontproperties=title_font)
    plt.axis('equal')

    if save_path:
        plt.savefig(os.path.join(save_path, 'security_events_pie.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 柱状图 - 使用水平柱状图避免标签重叠
    plt.figure(figsize=(12, 8))

    # 对事件类型进行换行处理
    wrapped_labels = wrap_labels(df['event_type'], 15)

    # 按计数排序
    df_sorted = df.sort_values('count', ascending=True)

    plt.barh(wrapped_labels, df_sorted['count'], color=sns.color_palette('viridis', len(df)))
    plt.title('安防事件类型统计', fontproperties=title_font)
    plt.xlabel('发生次数', fontproperties=medium_font)
    plt.ylabel('事件类型', fontproperties=medium_font)

    # 添加数据标签
    for i, v in enumerate(df_sorted['count']):
        plt.text(v + 0.5, i, str(v), va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'security_events_bar.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_device_usage_pattern(data, save_path=None):
    """可视化设备使用模式热力图"""
    if not data or len(data) == 0:
        print("没有可用的设备使用模式数据")
        return

    df = pd.DataFrame(data)

    # 如果设备太多，只显示前30个
    if len(df['device_name'].unique()) > 30:
        top_devices = df.groupby('device_name')['usage_count'].sum().nlargest(30).index
        df = df[df['device_name'].isin(top_devices)]
        print(f"设备数量过多，只显示前30个设备")

    pivot_df = df.pivot_table(index='device_name', columns='hour_of_day', values='usage_count', fill_value=0)

    # 对设备名称进行换行处理
    wrapped_index = wrap_labels(pivot_df.index, 12)
    pivot_df.index = wrapped_index

    plt.figure(figsize=(18, 14))  # 增大图像尺寸

    # 检查数据类型并动态设置fmt
    if pivot_df.values.dtype.kind in 'iu':  # 整数类型
        fmt = 'd'
    else:
        fmt = '.0f'  # 浮点数但显示为整数

    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=fmt, linewidths=.5,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    plt.title('设备使用模式热力图（按小时）', fontproperties=title_font)
    plt.xlabel('小时', fontproperties=medium_font)
    plt.ylabel('设备名称', fontproperties=medium_font)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'device_usage_pattern_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_energy_consumption(data, save_path=None):
    """可视化设备能耗分析"""
    if not data or len(data) == 0:
        print("没有可用的能耗数据")
        return

    df = pd.DataFrame(data)

    # 对设备类型进行换行处理
    df['wrapped_type'] = wrap_labels(df['device_type'], 15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 总使用时长饼图
    ax1.pie(
        df['total_hours'],
        labels=df['wrapped_type'],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel', len(df)),
        explode=[0.05] * len(df),
        textprops={'fontsize': 8}
    )
    ax1.set_title('各类设备总使用时长占比', fontproperties=title_font)

    # 平均使用时长柱状图
    sns.barplot(x='wrapped_type', y='avg_hours', data=df, palette='Set2', ax=ax2)
    ax2.set_title('各类设备平均单次使用时长', fontproperties=title_font)
    ax2.set_ylabel('小时', fontproperties=medium_font)
    ax2.set_xlabel('设备类型', fontproperties=medium_font)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数据标签
    for i, v in enumerate(df['avg_hours']):
        ax2.text(i, v + 0.05, f"{v:.1f}", ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'energy_consumption.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_concurrent_device_usage(data, save_path=None):
    """可视化设备同时使用关系图"""
    if not data or len(data) == 0:
        print("没有可用的设备同时使用数据")
        return

    # 如果组合太多，只显示前30个
    if len(data) > 30:
        data = sorted(data, key=lambda x: x['concurrent_count'], reverse=True)[:30]
        print(f"组合数量过多，只显示前30个组合")

    # 创建关系图
    plt.figure(figsize=(16, 12))
    G = nx.Graph()

    for item in data:
        G.add_edge(item['device1'], item['device2'], weight=item['concurrent_count'])

    if len(G.nodes) == 0:
        print("没有足够的节点创建网络图")
        return

    # 计算节点位置
    pos = nx.spring_layout(G, seed=42, k=0.4)  # 增加k值使节点更分散

    # 绘制节点和边
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_size = [d * 3000 / max_degree for d in degrees.values()]

    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_width = [w * 8 / max_weight for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='gray', alpha=0.5)

    # 节点标签换行处理
    labels = {node: '\n'.join(wrap(node, 12)) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family='sans-serif')

    # 添加边权重标签
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title('设备同时使用关系图', fontproperties=title_font)
    plt.axis('off')

    if save_path:
        plt.savefig(os.path.join(save_path, 'concurrent_device_usage_network.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_device_failure_rate(data, save_path=None):
    """可视化设备故障率"""
    if not data or len(data) == 0:
        print("没有可用的设备故障率数据")
        return

    df = pd.DataFrame(data)

    # 对设备类型进行换行处理
    df['wrapped_type'] = wrap_labels(df['device_type'], 15)

    plt.figure(figsize=(14, 8))
    sns.barplot(x='wrapped_type', y='failure_rate_percent', data=df, palette='rocket')
    plt.title('各类设备故障率', fontproperties=title_font)
    plt.xlabel('设备类型', fontproperties=medium_font)
    plt.ylabel('故障率 (%)', fontproperties=medium_font)

    # 添加数据标签
    for i, v in enumerate(df['failure_rate_percent']):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'device_failure_rate.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_feedback_sentiment(data, save_path=None):
    """可视化用户反馈情感分布"""
    if not data or len(data) == 0:
        print("没有可用的用户反馈情感数据")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))

    # 定义情感对应的颜色
    color_map = {
        'Positive': '#4CAF50',  # 绿色
        'Neutral': '#FFC107',  # 黄色
        'Negative': '#F44336'  # 红色
    }

    # 为每个情感类别分配颜色
    colors = [color_map.get(sentiment, '#999999') for sentiment in df['sentiment']]

    # 对情感标签进行换行处理
    wrapped_labels = wrap_labels(df['sentiment'], 10)

    plt.pie(
        df['count'],
        labels=wrapped_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        shadow=True,
        explode=[0.05] * len(df),
        textprops={'fontsize': 9}
    )
    plt.title('用户反馈情感分布', fontproperties=title_font)
    plt.axis('equal')

    if save_path:
        plt.savefig(os.path.join(save_path, 'feedback_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visualize_user_activity(data, save_path=None):
    """可视化用户活跃度"""
    if not data or len(data) == 0:
        print("没有可用的用户活跃度数据")
        return

    df = pd.DataFrame(data)

    # 对用户名进行换行处理
    df['wrapped_name'] = wrap_labels(df['user_name'], 12)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 用户使用次数排名
    df_sorted = df.sort_values('usage_count', ascending=False).head(10)
    if not df_sorted.empty:
        sns.barplot(x='usage_count', y='wrapped_name', data=df_sorted, palette='viridis', ax=ax1)
        ax1.set_title('用户设备使用次数排名 (Top 10)', fontproperties=title_font)
        ax1.set_xlabel('使用次数', fontproperties=medium_font)
        ax1.set_ylabel('用户名', fontproperties=medium_font)

        # 添加数据标签
        for i, v in enumerate(df_sorted['usage_count']):
            ax1.text(v + 5, i, str(v), va='center', fontsize=8)
    else:
        ax1.set_title('无数据', fontproperties=title_font)

    # 用户使用设备类型数量
    df_sorted_types = df.sort_values('device_types_used', ascending=False).head(10)
    if not df_sorted_types.empty:
        sns.barplot(x='device_types_used', y='wrapped_name', data=df_sorted_types, palette='mako', ax=ax2)
        ax2.set_title('用户使用设备类型数量 (Top 10)', fontproperties=title_font)
        ax2.set_xlabel('使用设备类型数量', fontproperties=medium_font)
        ax2.set_ylabel('')

        # 添加数据标签
        for i, v in enumerate(df_sorted_types['device_types_used']):
            ax2.text(v + 0.1, i, str(v), va='center', fontsize=8)
    else:
        ax2.set_title('无数据', fontproperties=title_font)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'user_activity.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 创建输出目录
    save_path = create_output_directory()
    print(f"所有可视化结果将保存到: {save_path}")

    # 映射端点名称到获取数据和可视化函数
    visualizations = {
        "device_usage_frequency": {
            "get_data": lambda: get_api_data("device_usage_frequency"),
            "visualize": visualize_device_usage_frequency
        },
        "concurrent_devices": {
            "get_data": lambda: get_api_data("concurrent_devices"),
            "visualize": visualize_concurrent_devices
        },
        "area_usage_impact": {
            "get_data": lambda: get_api_data("area_usage_impact"),
            "visualize": visualize_area_usage_impact
        },
        "security_event_stats": {
            "get_data": lambda: get_api_data("security_event_stats"),
            "visualize": visualize_security_event_stats
        },
        "device_usage_pattern": {
            "get_data": lambda: get_api_data("device_usage_pattern"),
            "visualize": visualize_device_usage_pattern
        },
        "energy_consumption": {
            "get_data": lambda: get_api_data("energy_consumption"),
            "visualize": visualize_energy_consumption
        },
        "concurrent_device_usage": {
            "get_data": lambda: get_api_data("concurrent_device_usage"),
            "visualize": visualize_concurrent_device_usage
        },
        "feedback_sentiment": {
            "get_data": lambda: get_api_data("feedback_sentiment"),
            "visualize": visualize_feedback_sentiment
        },
        "user_activity": {
            "get_data": lambda: get_api_data("user_activity"),
            "visualize": visualize_user_activity
        },
    }

    # 依次获取数据并进行可视化
    for name, viz in visualizations.items():
        print(f"\n正在处理: {name.replace('_', ' ').title()}...")
        data = viz["get_data"]()
        if data and len(data) > 0:
            print(f"获取到 {len(data)} 条记录")
            viz["visualize"](data, save_path)
            time.sleep(1)  # 避免连续请求过快
        else:
            print(f"无法获取 {name} 数据或数据为空，跳过")

    print(f"\n所有可视化已完成！结果保存在: {save_path}")


if __name__ == "__main__":
    # 检查API是否可用
    try:
        print(f"尝试连接到API服务: {BASE_URL}")
        response = requests.get(BASE_URL + "/health", timeout=10)
        if response.status_code == 200:
            print("API服务可用，开始可视化...")
            main()
        else:
            print(f"API服务不可用，状态码: {response.status_code}")
            print("请确保智能家居API服务正在运行")
    except requests.exceptions.RequestException as e:
        print(f"无法连接到API服务: {str(e)}")
        print("请检查:")
        print(f"1. API服务是否在 {BASE_URL} 运行")
        print("2. 网络连接是否正常")
        print("3. 防火墙设置是否允许连接")