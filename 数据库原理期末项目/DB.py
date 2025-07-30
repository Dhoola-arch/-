import sqlite3
from datetime import datetime
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
import uvicorn
import socket
from contextlib import closing
import logging
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartHomeAPI")

app = FastAPI(
    title="智能家居系统 API",
    description="提供智能家居设备管理、使用记录、安防事件和用户反馈的API接口",
    version="1.0.0",
    openapi_tags=[
        {"name": "用户管理", "description": "用户账户管理操作"},
        {"name": "房屋管理", "description": "房屋信息管理操作"},
        {"name": "设备管理", "description": "智能设备管理操作"},
        {"name": "设备使用记录", "description": "设备使用记录管理"},
        {"name": "安防事件", "description": "安全事件记录管理"},
        {"name": "用户反馈", "description": "用户反馈管理"},
        {"name": "分析功能", "description": "数据分析接口"},
        {"name": "可视化", "description": "数据可视化接口"},
        {"name": "系统管理", "description": "系统维护接口"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境需限制，调试时用 * 快速验证
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory="static"), name="static")




@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )




@app.get("/", tags=["系统管理"])
def home():
    """系统根端点"""
    return {
        "message": "Smart Home API is running!",
        "docs": "/docs",
        "redoc": "/redoc",
        "health_check": "/health"
    }


@app.get("/test", tags=["系统管理"])
def test_endpoint():
    """测试端点"""
    return {
        "status": "success",
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health_check": "/health",
            "reset_db": "/reset-db"
        }
    }


def find_free_port(start_port=8000, end_port=9000):
    """查找可用端口"""
    for port in range(start_port, end_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port available")


# 数据库初始化（添加索引）
def init_db():
    conn = sqlite3.connect('smart_home.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')

    # 创建房屋表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS houses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        address TEXT NOT NULL,
        area REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_houses_user ON houses(user_id)')

    # 创建设备表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        house_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (house_id) REFERENCES houses (id) ON DELETE CASCADE
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_devices_house ON devices(house_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_devices_type ON devices(type)')

    # 创建设备使用记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS device_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id INTEGER NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (device_id) REFERENCES devices (id) ON DELETE CASCADE
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_device ON device_usage(device_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_time ON device_usage(start_time)')

    # 创建安防事件表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS security_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        event_time TIMESTAMP NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (device_id) REFERENCES devices (id) ON DELETE CASCADE
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_device ON security_events(device_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON security_events(event_time)')

    # 创建用户反馈表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        rating INTEGER CHECK (rating BETWEEN 1 AND 5),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)')

    conn.commit()
    conn.close()

# 调用初始化函数
init_db()





# Pydantic 模型定义
class UserCreate(BaseModel):
    name: str
    email: str
    age: Optional[int] = None


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    created_at: datetime


class HouseCreate(BaseModel):
    user_id: int
    address: str
    area: float


class HouseResponse(BaseModel):
    id: int
    user_id: int
    address: str
    area: float
    created_at: datetime


class DeviceCreate(BaseModel):
    house_id: int
    name: str
    type: str


class DeviceResponse(BaseModel):
    id: int
    house_id: int
    name: str
    type: str
    created_at: datetime


class DeviceUsageCreate(BaseModel):
    device_id: int
    start_time: datetime
    end_time: datetime


class DeviceUsageResponse(BaseModel):
    id: int
    device_id: int
    start_time: datetime
    end_time: datetime
    created_at: datetime


class SecurityEventCreate(BaseModel):
    device_id: int
    event_type: str
    event_time: datetime
    description: Optional[str] = None


class SecurityEventResponse(BaseModel):
    id: int
    device_id: int
    event_type: str
    event_time: datetime
    description: Optional[str] = None
    created_at: datetime


class FeedbackCreate(BaseModel):
    user_id: int
    content: str
    rating: int


class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    content: str
    rating: int
    created_at: datetime


class DeviceUsageFrequency(BaseModel):
    device_id: int
    device_name: str
    frequency: int
    time_slot: str


class ConcurrentDevices(BaseModel):
    device1_id: int
    device1_name: str
    device2_id: int
    device2_name: str
    concurrent_count: int


class AreaUsageImpact(BaseModel):
    area_range: str
    device_type: str
    avg_duration: float
    usage_count: int


class SecurityEventStats(BaseModel):
    event_type: str
    count: int
    percentage: float


class VisualizationResponse(BaseModel):
    plot_base64: str
    description: str


# 安全的数据库连接函数
def get_db_connection():
    conn = sqlite3.connect('smart_home.db', detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(query: str, params: tuple = (), fetch_one: bool = False, commit: bool = False):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if commit:
            conn.commit()

        if fetch_one:
            result = cursor.fetchone()
        else:
            result = cursor.fetchall()
        return result
    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database operation failed")
    finally:
        conn.close()


def populate_sample_data():
    """填充更全面的示例数据，支持多样化的可视化演示"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 检查是否已有数据
        if cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            logger.info("数据库已有数据，跳过示例数据填充")
            return

        logger.info("开始填充更全面的示例数据...")

        # 创建更多示例用户（20个用户，覆盖不同年龄段和使用习惯）
        users = [
            ("张三", "zhangsan@example.com", 35),  # 中年用户，家庭稳定
            ("李四", "lisi@example.com", 28),  # 青年用户，科技爱好者
            ("王五", "wangwu@example.com", 42),  # 中老年用户，注重节能
            ("赵六", "zhaoliu@example.com", 31),  # 职场人士，使用时间规律
            ("钱七", "qianqi@example.com", 26),  # 年轻用户，夜间活动频繁
            ("孙八", "sunba@example.com", 55),  # 老年用户，使用较少
            ("周九", "zhoujiu@example.com", 22),  # 学生，使用不规律
            ("吴十", "wushi@example.com", 38),  # 家庭主妇，日间使用多
            ("郑十一", "zhengshiyi@example.com", 45),  # 企业主，安全意识强
            ("陈十二", "chenshier@example.com", 30),  # 自由职业者，全天候使用
            ("林十三", "linshisan@example.com", 29),  # 技术极客，频繁使用
            ("黄十四", "huangshisi@example.com", 33),  # 环保主义者，节能使用
            ("刘十五", "liushiwu@example.com", 48),  # 忙碌高管，很少在家
            ("马十六", "mashiliu@example.com", 27),  # 游戏玩家，娱乐设备多
            ("朱十七", "zhushiqi@example.com", 36),  # 家庭主夫，厨房设备多
            ("胡十八", "hushiba@example.com", 41),  # 安全专家，安防设备多
            ("高十九", "gaoshijiu@example.com", 53),  # 退休教师，使用简单
            ("谢二十", "xieershi@example.com", 24),  # 租房者，设备少
            ("罗二十一", "luoershiyi@example.com", 39),  # 技术顾问，设备齐全
            ("梁二十二", "liangershier@example.com", 32)  # 新晋父母，关注安全
        ]
        user_ids = []
        for user in users:
            cursor.execute(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                user
            )
            user_ids.append(cursor.lastrowid)

        # 创建多样化的房屋（不同面积、位置、装修风格）
        houses = [
            (user_ids[0], "北京市海淀区中关村大街1号", 120.5),  # 城市中心大户型
            (user_ids[1], "上海市浦东新区张江高科技园区", 89.3),  # 青年公寓
            (user_ids[2], "广州市天河区珠江新城", 150.2),  # 豪华住宅
            (user_ids[3], "深圳市南山区科技园", 75.8),  # 小户型
            (user_ids[4], "杭州市西湖区文三路", 110.0),  # 中等户型
            (user_ids[5], "成都市锦江区春熙路", 90.7),  # 退休公寓
            (user_ids[6], "南京市鼓楼区", 65.2),  # 学生公寓
            (user_ids[7], "武汉市武昌区", 130.4),  # 家庭住宅
            (user_ids[8], "重庆市渝中区", 140.9),  # 安全要求高的住宅
            (user_ids[9], "天津市和平区", 100.3),  # 现代风格住宅
            (user_ids[10], "西安市雁塔区", 95.7),   # 技术爱好者住宅
            (user_ids[11], "长沙市岳麓区", 85.4),   # 环保住宅
            (user_ids[12], "厦门市思明区", 200.0),  # 豪华别墅
            (user_ids[13], "青岛市市南区", 70.3),   # 海滨公寓
            (user_ids[14], "苏州市工业园区", 115.8), # 家庭住宅
            (user_ids[15], "大连市中山区", 125.6),  # 安全示范住宅
            (user_ids[16], "佛山市禅城区", 88.9),   # 退休社区
            (user_ids[17], "合肥市蜀山区", 55.2),   # 出租公寓
            (user_ids[18], "宁波市鄞州区", 180.3),  # 智能家居展示住宅
            (user_ids[19], "东莞市南城区", 92.1)    # 年轻家庭住宅
        ]
        house_ids = []
        for house in houses:
            cursor.execute(
                "INSERT INTO houses (user_id, address, area) VALUES (?, ?, ?)",
                (house[0], house[1], house[2])
            )
            house_ids.append(cursor.lastrowid)

        device_categories = {
            "照明": ["客厅主灯", "卧室灯", "走廊灯", "厨房灯", "书房灯", "浴室灯"],
            "安防": ["大门摄像头", "客厅监控", "车库传感器", "门窗传感器", "烟雾报警器", "水浸传感器"],
            "环境": ["空调", "加湿器", "空气净化器", "除湿机", "新风系统", "智能窗帘"],
            "娱乐": ["智能电视", "音响系统", "投影仪", "游戏机", "家庭影院"],
            "厨房": ["智能冰箱", "烤箱", "咖啡机", "洗碗机", "微波炉", "电饭煲"],
            "家电": ["洗衣机", "烘干机", "扫地机器人", "吸尘器", "热水器", "智能门锁"]
        }

        # 创建设备（为每个房屋分配不同数量和类型的设备，模拟真实场景）
        device_category_names = list(device_categories.keys())
        device_ids = []
        device_type_map = {}  # 设备ID到类型的映射
        device_house_map = {}  # 设备ID到房屋ID的映射

        for i, house_id in enumerate(house_ids):
            # 根据房屋面积和用户类型分配不同设备
            if i < 5:  # 大户型房屋
                for device_type, names in device_categories.items():
                    for name in names[:4]:  # 每种类型选4个设备
                        cursor.execute(
                            "INSERT INTO devices (house_id, name, type) VALUES (?, ?, ?)",
                            (house_id, f"{name}_{house_id}", device_type)
                        )
                        device_id = cursor.lastrowid
                        device_ids.append(device_id)
                        device_type_map[device_id] = device_type
                        device_house_map[device_id] = house_id
            elif i < 10:  # 中等户型
                for device_type, names in device_categories.items():
                    for name in names[:3]:  # 每种类型选3个设备
                        cursor.execute(
                            "INSERT INTO devices (house_id, name, type) VALUES (?, ?, ?)",
                            (house_id, f"{name}_{house_id}", device_type)
                        )
                        device_id = cursor.lastrowid
                        device_ids.append(device_id)
                        device_type_map[device_id] = device_type
                        device_house_map[device_id] = house_id
            elif i < 15:  # 小户型
                for device_type, names in device_categories.items():
                    if device_type in ["照明", "安防", "环境", "家电"]:  # 小户型更关注基础设备
                        for name in names[:2]:  # 每种类型选2个设备
                            cursor.execute(
                                "INSERT INTO devices (house_id, name, type) VALUES (?, ?, ?)",
                                (house_id, f"{name}_{house_id}", device_type)
                            )
                            device_id = cursor.lastrowid
                            device_ids.append(device_id)
                            device_type_map[device_id] = device_type
                            device_house_map[device_id] = house_id
            else:  # 特殊户型
                for device_type, names in device_categories.items():
                    if device_type in ["照明", "安防", "厨房"]:  # 特殊需求
                        for name in names[:3]:  # 每种类型选3个设备
                            cursor.execute(
                                "INSERT INTO devices (house_id, name, type) VALUES (?, ?, ?)",
                                (house_id, f"{name}_{house_id}", device_type)
                            )
                            device_id = cursor.lastrowid
                            device_ids.append(device_id)
                            device_type_map[device_id] = device_type
                            device_house_map[device_id] = house_id

        # 创建设备使用记录（更复杂的时间模式，模拟真实使用场景）
        import random
        from datetime import datetime, timedelta, time

        # 定义不同设备类型的使用模式
        device_usage_patterns = {
            "照明": {
                "morning": (6, 10, 0.3),  # 早上使用概率30%
                "afternoon": (12, 18, 0.1),  # 下午使用概率10%
                "evening": (18, 23, 0.8),  # 晚上使用概率80%
                "night": (23, 6, 0.2),  # 夜间使用概率20%
                "duration": (0.5, 4)  # 使用时长0.5-4小时
            },
            "安防": {
                "morning": (6, 10, 0.1),
                "afternoon": (12, 18, 0.1),
                "evening": (18, 23, 0.1),
                "night": (23, 6, 0.9),  # 夜间安防设备高概率工作
                "duration": (8, 12)  # 安防设备持续工作时间长
            },
            "环境": {
                "morning": (6, 10, 0.6),
                "afternoon": (12, 18, 0.7),  # 下午环境设备使用频繁
                "evening": (18, 23, 0.6),
                "night": (23, 6, 0.3),
                "duration": (2, 6)
            },
            "娱乐": {
                "morning": (6, 10, 0.1),
                "afternoon": (12, 18, 0.2),
                "evening": (18, 23, 0.7),  # 晚上娱乐设备使用频繁
                "night": (23, 6, 0.3),
                "duration": (1, 5)
            },
            "厨房": {
                "morning": (6, 10, 0.7),  # 早餐时间
                "afternoon": (12, 14, 0.8),  # 午餐时间
                "evening": (18, 20, 0.8),  # 晚餐时间
                "night": (23, 6, 0.05),
                "duration": (0.5, 2)
            },
            "家电": {
                "morning": (6, 10, 0.4),
                "afternoon": (12, 18, 0.6),  # 下午做家务
                "evening": (18, 23, 0.4),
                "night": (23, 6, 0.05),
                "duration": (1, 4)
            }
        }

        # 为每个设备创建使用记录
        for device_id in device_ids:
            if device_id not in device_type_map:
                logger.warning(f"设备ID {device_id} 类型信息缺失，跳过使用记录生成")
                continue

            device_type = device_type_map[device_id]
            house_id = device_house_map[device_id]  # 使用预存的房屋ID

            # 基于房屋所属用户类型调整使用频率
            user_type = house_id % 4  # 将用户分为4类（0,1,2,3）
            base_usage_days = {
                0: 90,  # 活跃用户
                1: 60,  # 普通用户
                2: 30,  # 低频用户
                3: 15   # 非常低频用户
            }
            usage_days = base_usage_days.get(user_type, 60)

            # 为每个设备创建使用记录
            for day_offset in range(random.randint(usage_days//2, usage_days)):
                day = datetime.now() - timedelta(days=day_offset)

                # 获取当前设备类型的使用模式
                usage_pattern = device_usage_patterns[device_type]

                # 根据时间段决定是否使用设备
                for slot_name, slot_info in [
                    ("morning", usage_pattern["morning"]),
                    ("afternoon", usage_pattern["afternoon"]),
                    ("evening", usage_pattern["evening"]),
                    ("night", usage_pattern["night"])
                ]:
                    start_hour, end_hour, probability = slot_info

                    # 处理跨天的时间范围（如23点到次日6点）
                    if start_hour > end_hour:
                        # 跨天情况：拆分为两段时间
                        # 第一段：start_hour 到 23点59分
                        if random.random() < probability:
                            start_h = random.randint(start_hour, 23)
                            start_m = random.randint(0, 59)
                            start_time = day.replace(hour=start_h, minute=start_m)

                            min_dur, max_dur = usage_pattern["duration"]
                            duration = random.uniform(min_dur, max_dur)
                            end_time = start_time + timedelta(hours=duration)

                            # 确保结束时间不超过当天23:59
                            if end_time.day > start_time.day:
                                end_time = start_time.replace(hour=23, minute=59)

                            cursor.execute(
                                "INSERT INTO device_usage (device_id, start_time, end_time) VALUES (?, ?, ?)",
                                (device_id, start_time, end_time)
                            )

                        # 第二段：0点到 end_hour
                        if random.random() < probability:
                            start_h = random.randint(0, end_hour - 1)
                            start_m = random.randint(0, 59)
                            # 注意这里是次日，需要加一天
                            start_time = (day + timedelta(days=1)).replace(hour=start_h, minute=start_m)

                            min_dur, max_dur = usage_pattern["duration"]
                            duration = random.uniform(min_dur, max_dur)
                            end_time = start_time + timedelta(hours=duration)

                            cursor.execute(
                                "INSERT INTO device_usage (device_id, start_time, end_time) VALUES (?, ?, ?)",
                                (device_id, start_time, end_time)
                            )

                    else:
                        # 普通情况：直接使用给定的时间范围
                        if random.random() < probability:
                            start_h = random.randint(start_hour, end_hour - 1)
                            start_m = random.randint(0, 59)
                            start_time = day.replace(hour=start_h, minute=start_m)

                            min_dur, max_dur = usage_pattern["duration"]
                            duration = random.uniform(min_dur, max_dur)
                            end_time = start_time + timedelta(hours=duration)

                            # 避免结束时间跨天
                            if end_time.day != start_time.day:
                                end_time = start_time + timedelta(hours=min_dur)

                            cursor.execute(
                                "INSERT INTO device_usage (device_id, start_time, end_time) VALUES (?, ?, ?)",
                                (device_id, start_time, end_time)
                            )

        # 创建安防事件（更丰富的事件类型和时间分布）
        event_types = [
            "门禁异常", "移动检测", "火灾警报", "水浸检测", "闯入警报",
            "设备离线", "温度异常", "烟雾报警", "玻璃破碎", "非法访问",
            "电池电量低", "传感器故障", "网络断开", "固件错误", "维护提醒"  # 新增故障相关事件
        ]

        # 为不同设备类型设置不同的事件概率
        device_event_probability = {
            "安防": 0.7,  # 安防设备更容易触发事件
            "环境": 0.4,
            "照明": 0.2,
            "娱乐": 0.3,
            "厨房": 0.5,  # 厨房设备可能触发火灾、水浸等
            "家电": 0.4
        }

        # 设备故障率映射（新增）
        device_failure_rates = {
            "安防": 0.15,
            "环境": 0.10,
            "照明": 0.05,
            "娱乐": 0.12,
            "厨房": 0.20,  # 厨房设备故障率较高
            "家电": 0.18
        }

        for day_offset in range(120):  # 过去120天的事件
            day = datetime.now() - timedelta(days=day_offset)

            # 每天随机选择一些设备触发事件（确保只选择有效的设备ID）
            valid_device_ids = [id for id in device_ids if id in device_type_map]
            if not valid_device_ids:
                continue

            # 每天事件数量（1-10个）
            for _ in range(random.randint(1, 10)):
                device_id = random.choice(valid_device_ids)
                device_type = device_type_map[device_id]

                # 根据设备类型决定事件概率
                if random.random() < device_event_probability.get(device_type, 0.2):
                    # 事件时间（集中在夜间和无人时段）
                    if random.random() < 0.7:  # 70%的事件发生在夜间或无人时段
                        event_hour = random.randint(0, 6) or random.randint(12, 14)  # 夜间或午休时间
                    else:
                        event_hour = random.randint(7, 23)

                    event_minute = random.randint(0, 59)
                    event_time = day.replace(hour=event_hour, minute=event_minute)

                    # 根据设备故障率决定是否为故障事件
                    is_failure = random.random() < device_failure_rates.get(device_type, 0.1)

                    if is_failure:
                        # 故障相关事件
                        failure_events = ["设备离线", "传感器故障", "网络断开", "固件错误", "电池电量低"]
                        event_type = random.choice(failure_events)
                    else:
                        # 普通安全事件
                        event_type = random.choice(event_types)

                    # 根据事件类型生成不同的描述
                    descriptions = {
                        "门禁异常": ["未授权尝试开门", "密码连续输入错误", "门未正常关闭"],
                        "移动检测": ["检测到异常移动", "夜间有人活动", "触发区域移动警报"],
                        "火灾警报": ["烟雾浓度过高", "温度异常升高", "检测到明火"],
                        "水浸检测": ["厨房漏水", "浴室积水", "水管破裂"],
                        "闯入警报": ["窗户被撬动", "检测到非法入侵", "安防系统触发"],
                        "设备离线": ["设备信号丢失", "网络连接中断", "电池电量过低"],
                        "温度异常": ["室内温度过高", "空调故障导致温度异常", "温控器失灵"],
                        "烟雾报警": ["烹饪产生大量烟雾", "吸烟触发警报", "电器冒烟"],
                        "玻璃破碎": ["检测到玻璃破碎声音", "窗户安全系统触发", "疑似暴力闯入"],
                        "非法访问": ["非授权设备尝试连接", "密码被破解", "系统检测到异常登录"],
                        "电池电量低": ["设备电量不足10%", "需要更换电池", "低电量警告"],
                        "传感器故障": ["传感器读数异常", "校准失败", "硬件故障"],
                        "网络断开": ["Wi-Fi连接丢失", "网络不稳定", "路由器故障"],
                        "固件错误": ["固件更新失败", "系统崩溃", "需要重启设备"],
                        "维护提醒": ["需要清洁维护", "滤网更换提醒", "系统自检失败"]
                    }

                    description = random.choice(descriptions.get(event_type, ["系统自动报警"]))

                    cursor.execute(
                        "INSERT INTO security_events (device_id, event_type, event_time, description) VALUES (?, ?, ?, ?)",
                        (device_id, event_type, event_time, description)
                    )

        # 创建用户反馈（更丰富的反馈内容和评分分布）
        feedback_contents = [
            "系统非常好用，推荐！",
            "设备偶尔会掉线，希望优化连接稳定性",
            "界面设计很直观，但某些功能位置不太好找",
            "希望能增加更多设备支持，尤其是厨房电器",
            "报警系统很灵敏，给我带来了安全感",
            "手机APP需要优化，偶尔会卡顿",
            "客服响应迅速，问题解决很及时",
            "安装过程很顺利，技术人员很专业",
            "希望增加自动化场景，比如离家模式一键启动",
            "能耗统计功能很有用，但数据展示可以更直观",
            "安防摄像头画质清晰，但夜视效果一般",
            "智能门锁指纹识别速度有待提高",
            "环境监测数据不准确，希望校准算法",
            "语音控制功能很好用，但识别率需要提升",
            "价格偏高，希望推出更多套餐优惠",
            "系统稳定运行三个月无故障，非常满意",
            "设备响应速度快，用户体验很好",
            "智能场景设置复杂，需要简化",
            "设备兼容性问题，部分设备无法接入",
            "能效管理功能节省了电费，很实用",
            "安全功能过于敏感，经常误报",
            "设备固件更新后出现新问题",
            "用户手册不详细，需要更多使用指南",
            "多用户管理功能不够完善",
            "远程控制功能非常方便，随时随地管理家居"
        ]

        # 为每个用户创建1-5条反馈，评分更分散
        for user_id in user_ids:
            for _ in range(random.randint(1, 5)):
                # 评分分布：40%概率给5分，20%概率给4分，20%概率给3分，10%概率给2分，10%概率给1分
                weights = [0.4, 0.2, 0.2, 0.1, 0.1]
                rating = random.choices([5, 4, 3, 2, 1], weights=weights)[0]
                content = random.choice(feedback_contents)

                # 根据评分调整反馈内容的情感倾向
                if rating >= 4:
                    positive_comments = ["非常满意", "超出预期", "推荐购买", "很实用", "值得投资"]
                    content = f"{random.choice(positive_comments)}，{content}"
                elif rating == 3:
                    neutral_comments = ["总体还行", "基本满意", "中规中矩", "有待改进", "可以接受"]
                    content = f"{random.choice(neutral_comments)}，{content}"
                else:
                    negative_comments = ["不太满意", "体验较差", "存在问题", "需要改进", "不推荐"]
                    content = f"{random.choice(negative_comments)}，{content}"

                # 添加一些具体设备相关的反馈
                if random.random() > 0.7:  # 30%的反馈提及具体设备
                    # 使用之前保存的设备类别列表
                    device_category = random.choice(device_category_names)
                    device_comments = {
                        "照明": ["灯光调节不够灵活", "亮度自动调节很智能", "色温范围有限"],
                        "安防": ["摄像头角度有限", "报警响应及时", "夜视效果不佳"],
                        "环境": ["温控精度高", "空气质量监测准确", "噪音有点大"],
                        "娱乐": ["音响效果震撼", "投屏功能不稳定", "内容资源丰富"],
                        "厨房": ["冰箱智能管理实用", "烤箱温度控制精准", "咖啡机清洗麻烦"],
                        "家电": ["洗衣机洗得很干净", "烘干机耗电量大", "扫地机器人避障能力强"]
                    }
                    device_comment = random.choice(device_comments.get(device_category, [""]))
                    content = f"{content} 特别是{device_category}设备，{device_comment}"

                cursor.execute(
                    "INSERT INTO feedback (user_id, content, rating) VALUES (?, ?, ?)",
                    (user_id, content, rating)
                )

        conn.commit()
        logger.info("全面示例数据填充完成！")

    except sqlite3.Error as e:
        logger.error(f"填充示例数据失败: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

# 调用填充函数
populate_sample_data()

# 用户管理 API
@app.post("/users/", response_model=UserResponse, tags=["用户管理"])
def create_user(user: UserCreate):
    try:
        result = execute_query(
            "INSERT INTO users (name, email, age) VALUES (?, ?, ?) RETURNING id, name, email, age, created_at",
            (user.name, user.email, user.age),
            fetch_one=True,
            commit=True
        )
        return dict(result)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already exists")


@app.get("/users/{user_id}", response_model=UserResponse, tags=["用户管理"])
def get_user(user_id: int):
    result = execute_query(
        "SELECT id, name, email, age, created_at FROM users WHERE id = ?",
        (user_id,),
        fetch_one=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(result)


@app.delete("/users/{user_id}", tags=["用户管理"])
def delete_user(user_id: int):
    execute_query(
        "DELETE FROM users WHERE id = ?",
        (user_id,),
        commit=True
    )
    return {"message": "User deleted successfully"}


# 房屋管理 API
@app.post("/houses/", response_model=HouseResponse, tags=["房屋管理"])
def create_house(house: HouseCreate):
    # 验证用户存在
    user_exists = execute_query(
        "SELECT 1 FROM users WHERE id = ?",
        (house.user_id,),
        fetch_one=True
    )
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")

    result = execute_query(
        "INSERT INTO houses (user_id, address, area) VALUES (?, ?, ?) RETURNING id, user_id, address, area, created_at",
        (house.user_id, house.address, house.area),
        fetch_one=True,
        commit=True
    )
    return dict(result)


@app.get("/houses/{house_id}", response_model=HouseResponse, tags=["房屋管理"])
def get_house(house_id: int):
    result = execute_query(
        "SELECT id, user_id, address, area, created_at FROM houses WHERE id = ?",
        (house_id,),
        fetch_one=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="House not found")
    return dict(result)


# 设备管理 API
@app.post("/devices/", response_model=DeviceResponse, tags=["设备管理"])
def create_device(device: DeviceCreate):
    # 验证房屋存在
    house_exists = execute_query(
        "SELECT 1 FROM houses WHERE id = ?",
        (device.house_id,),
        fetch_one=True
    )
    if not house_exists:
        raise HTTPException(status_code=404, detail="House not found")

    result = execute_query(
        "INSERT INTO devices (house_id, name, type) VALUES (?, ?, ?) RETURNING id, house_id, name, type, created_at",
        (device.house_id, device.name, device.type),
        fetch_one=True,
        commit=True
    )
    return dict(result)


@app.get("/devices/{device_id}", response_model=DeviceResponse, tags=["设备管理"])
def get_device(device_id: int):
    result = execute_query(
        "SELECT id, house_id, name, type, created_at FROM devices WHERE id = ?",
        (device_id,),
        fetch_one=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Device not found")
    return dict(result)


# 设备使用记录 API
@app.post("/device-usage/", response_model=DeviceUsageResponse, tags=["设备使用记录"])
def create_device_usage(usage: DeviceUsageCreate):
    if usage.end_time <= usage.start_time:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    # 验证设备存在
    device_exists = execute_query(
        "SELECT 1 FROM devices WHERE id = ?",
        (usage.device_id,),
        fetch_one=True
    )
    if not device_exists:
        raise HTTPException(status_code=404, detail="Device not found")

    result = execute_query(
        "INSERT INTO device_usage (device_id, start_time, end_time) VALUES (?, ?, ?) RETURNING id, device_id, start_time, end_time, created_at",
        (usage.device_id, usage.start_time, usage.end_time),
        fetch_one=True,
        commit=True
    )
    return dict(result)


# 安防事件 API
@app.post("/security-events/", response_model=SecurityEventResponse, tags=["安防事件"])
def create_security_event(event: SecurityEventCreate):
    # 验证设备存在
    device_exists = execute_query(
        "SELECT 1 FROM devices WHERE id = ?",
        (event.device_id,),
        fetch_one=True
    )
    if not device_exists:
        raise HTTPException(status_code=404, detail="Device not found")

    result = execute_query(
        "INSERT INTO security_events (device_id, event_type, event_time, description) VALUES (?, ?, ?, ?) RETURNING id, device_id, event_type, event_time, description, created_at",
        (event.device_id, event.event_type, event.event_time, event.description),
        fetch_one=True,
        commit=True
    )
    return dict(result)


# 用户反馈 API
@app.post("/feedback/", response_model=FeedbackResponse, tags=["用户反馈"])
def create_feedback(feedback: FeedbackCreate):
    if feedback.rating < 1 or feedback.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    # 验证用户存在
    user_exists = execute_query(
        "SELECT 1 FROM users WHERE id = ?",
        (feedback.user_id,),
        fetch_one=True
    )
    if not user_exists:
        raise HTTPException(status_code=404, detail="User not found")

    result = execute_query(
        "INSERT INTO feedback (user_id, content, rating) VALUES (?, ?, ?) RETURNING id, user_id, content, rating, created_at",
        (feedback.user_id, feedback.content, feedback.rating),
        fetch_one=True,
        commit=True
    )
    return dict(result)


# 分析功能 API
@app.get("/analysis/device-usage-frequency/", response_model=List[DeviceUsageFrequency], tags=["分析功能"])
def get_device_usage_frequency():
    """分析不同设备的使用频率和使用时间段"""
    query = """
    SELECT 
        d.id AS device_id,
        d.name AS device_name,
        COUNT(*) AS frequency,
        CASE 
            WHEN CAST(strftime('%H', start_time) AS INTEGER) BETWEEN 6 AND 12 THEN 'Morning (6-12)'
            WHEN CAST(strftime('%H', start_time) AS INTEGER) BETWEEN 12 AND 18 THEN 'Afternoon (12-18)'
            WHEN CAST(strftime('%H', start_time) AS INTEGER) BETWEEN 18 AND 24 THEN 'Evening (18-24)'
            ELSE 'Night (0-6)'
        END AS time_slot
    FROM device_usage du
    JOIN devices d ON du.device_id = d.id
    GROUP BY d.id, time_slot
    ORDER BY frequency DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []


@app.get("/analysis/concurrent-devices/", response_model=List[ConcurrentDevices], tags=["分析功能"])
def get_concurrent_devices():
    """找出经常同时使用的设备"""
    query = """
    SELECT 
        d1.id AS device1_id,
        d1.name AS device1_name,
        d2.id AS device2_id,
        d2.name AS device2_name,
        COUNT(*) AS concurrent_count
    FROM device_usage du1
    JOIN device_usage du2 
        ON du1.device_id < du2.device_id
        AND du1.start_time <= du2.end_time
        AND du1.end_time >= du2.start_time
    JOIN devices d1 ON du1.device_id = d1.id
    JOIN devices d2 ON du2.device_id = d2.id
    GROUP BY d1.id, d2.id
    HAVING concurrent_count > 2
    ORDER BY concurrent_count DESC
    LIMIT 10
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []


@app.get("/analysis/area-usage-impact/", response_model=List[AreaUsageImpact], tags=["分析功能"])
def get_area_usage_impact():
    """分析房屋面积对设备使用行为的影响"""
    query = """
    SELECT 
        CASE 
            WHEN h.area < 50 THEN '0-50'
            WHEN h.area BETWEEN 50 AND 100 THEN '50-100'
            WHEN h.area BETWEEN 100 AND 150 THEN '100-150'
            ELSE '150+'
        END AS area_range,
        d.type AS device_type,
        AVG((JULIANDAY(du.end_time) - JULIANDAY(du.start_time)) * 1440) AS avg_duration,
        COUNT(*) AS usage_count
    FROM device_usage du
    JOIN devices d ON du.device_id = d.id
    JOIN houses h ON d.house_id = h.id
    GROUP BY area_range, device_type
    ORDER BY area_range, usage_count DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []

@app.get("/analysis/security-event-stats/", response_model=List[SecurityEventStats], tags=["分析功能"])
def get_security_event_stats():
    """分析安防事件类型分布"""
    query = """
    SELECT 
        event_type,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM security_events), 2) AS percentage
    FROM security_events
    GROUP BY event_type
    ORDER BY count DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []


# 可视化 API
@app.get("/visualization/device-usage-frequency/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_device_usage_frequency():
    """可视化设备使用频率"""
    data = get_device_usage_frequency()
    if not data:
        raise HTTPException(status_code=404, detail="No data available for visualization")

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index='device_name', columns='time_slot', values='frequency', fill_value=0)

    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', stacked=True)
    plt.title('设备使用频率按时间段分布')
    plt.ylabel('使用次数')
    plt.xlabel('设备名称')
    plt.xticks(rotation=45)
    plt.legend(title='时间段')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "各设备在不同时间段的使用频率分布"
    }


@app.get("/visualization/area-usage-impact/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_area_usage_impact():
    """可视化房屋面积对设备使用行为的影响"""
    data = get_area_usage_impact()
    if not data:
        raise HTTPException(status_code=404, detail="No data available for visualization")

    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 8))
    for device_type in df['device_type'].unique():
        subset = df[df['device_type'] == device_type]
        plt.plot(subset['area_range'], subset['avg_duration'], marker='o', label=device_type)

    plt.title('不同房屋面积下各类设备的平均使用时长')
    plt.ylabel('平均使用时长 (分钟)')
    plt.xlabel('房屋面积范围 (平方米)')
    plt.legend(title='设备类型')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "不同房屋面积下各类设备的平均使用时长"
    }


@app.get("/visualization/security-event-stats/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_security_event_stats():
    """可视化安防事件类型分布"""
    data = get_security_event_stats()
    if not data:
        raise HTTPException(status_code=404, detail="No data available for visualization")

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    plt.pie(df['count'], labels=df['event_type'], autopct='%1.1f%%',
            startangle=90, shadow=True, explode=[0.05] * len(df))
    plt.title('安防事件类型分布')
    plt.axis('equal')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "安防事件类型分布饼图"
    }

# 分析功能API端点
@app.get("/analysis/device-usage-pattern/", response_model=List[dict], tags=["分析功能"])
def get_device_usage_pattern():
    """分析设备使用模式（按小时）"""
    query = """
    SELECT 
        d.name AS device_name,
        CAST(strftime('%H', start_time) AS INTEGER) AS hour_of_day,
        COUNT(*) AS usage_count
    FROM device_usage du
    JOIN devices d ON du.device_id = d.id
    GROUP BY d.name, hour_of_day
    ORDER BY hour_of_day, device_name
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []

@app.get("/analysis/energy-consumption/", response_model=List[dict], tags=["分析功能"])
def get_energy_consumption():
    """分析设备能耗（基于使用时长）"""
    query = """
    SELECT 
        d.type AS device_type,
        ROUND(SUM(JULIANDAY(du.end_time) - JULIANDAY(du.start_time)) * 24, 2) AS total_hours,
        COUNT(*) AS usage_count,
        ROUND(AVG(JULIANDAY(du.end_time) - JULIANDAY(du.start_time)) * 24, 2) AS avg_hours
    FROM device_usage du
    JOIN devices d ON du.device_id = d.id
    GROUP BY d.type
    ORDER BY total_hours DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []

@app.get("/analysis/concurrent-device-usage/", response_model=List[dict], tags=["分析功能"])
def get_concurrent_device_usage():
    """分析设备同时使用情况"""
    query = """
    SELECT 
        d1.name AS device1,
        d2.name AS device2,
        COUNT(*) AS concurrent_count
    FROM device_usage du1
    JOIN device_usage du2 
        ON du1.start_time <= du2.end_time 
        AND du1.end_time >= du2.start_time 
        AND du1.device_id < du2.device_id
    JOIN devices d1 ON du1.device_id = d1.id
    JOIN devices d2 ON du2.device_id = d2.id
    GROUP BY d1.name, d2.name
    HAVING concurrent_count > 5
    ORDER BY concurrent_count DESC
    LIMIT 10
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []


# 设备使用模式热力图
@app.get("/visualization/device-usage-pattern/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_device_usage_pattern():
    """可视化设备使用模式热力图"""
    data = get_device_usage_pattern()

    if not data:
        raise HTTPException(status_code=404, detail="No data available")

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(
        index='device_name',
        columns='hour_of_day',
        values='usage_count',
        fill_value=0
    )

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt="d", linewidths=.5)
    plt.title('设备使用模式热力图（按小时）')
    plt.xlabel('小时')
    plt.ylabel('设备名称')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "各设备在不同小时的使用频率热力图"
    }


# 设备能耗分析
@app.get("/visualization/energy-consumption/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_energy_consumption():
    """可视化设备能耗分析"""
    data = get_energy_consumption()

    if not data:
        raise HTTPException(status_code=404, detail="No data available")

    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 总使用时长饼图
    ax1.pie(
        df['total_hours'],
        labels=df['device_type'],
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('pastel'),
        explode=[0.05] * len(df)
    )
    ax1.set_title('各类设备总使用时长占比')

    # 平均使用时长柱状图
    ax2.bar(
        df['device_type'],
        df['avg_hours'],
        color=sns.color_palette('Set2')
    )
    ax2.set_title('各类设备平均单次使用时长')
    ax2.set_ylabel('小时')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "设备能耗分析：总使用时长占比和平均单次使用时长"
    }


# 设备同时使用关系图
@app.get("/visualization/concurrent-device-usage/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_concurrent_device_usage():
    """可视化设备同时使用关系"""
    data = get_concurrent_device_usage()

    if not data:
        raise HTTPException(status_code=404, detail="No data available")

    df = pd.DataFrame(data)

    # 创建关系图
    plt.figure(figsize=(12, 8))

    # 创建节点和边
    nodes = set(df['device1'].tolist() + df['device2'].tolist())
    edges = [(row['device1'], row['device2'], row['concurrent_count']) for _, row in df.iterrows()]

    # 创建图
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    # 计算节点位置
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点和边
    node_size = [d * 100 for d in dict(G.degree()).values()]
    edge_width = [d['weight'] * 0.1 for u, v, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # 添加边权重标签
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title('设备同时使用关系图')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "设备同时使用关系网络图（线宽表示同时使用频率）"
    }



@app.get("/visualization/concurrent-devices/", response_model=VisualizationResponse, tags=["可视化"])
def visualize_concurrent_devices():
    """可视化经常同时使用的设备"""
    data = get_concurrent_devices()
    if not data:
        raise HTTPException(status_code=404, detail="No data available for visualization")

    df = pd.DataFrame(data)
    df['device_pair'] = df['device1_name'] + ' & ' + df['device2_name']

    plt.figure(figsize=(12, 8))
    plt.barh(df['device_pair'], df['concurrent_count'], color='skyblue')
    plt.title('经常同时使用的设备组合')
    plt.xlabel('同时使用次数')
    plt.ylabel('设备组合')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "plot_base64": plot_base64,
        "description": "经常同时使用的设备组合及其同时使用次数"
    }


@app.get("/analysis/user-activity/", response_model=List[dict], tags=["分析功能"])
def get_user_activity():
    """分析用户活跃度（按设备使用次数）"""
    query = """
    SELECT 
        u.id AS user_id,
        u.name AS user_name,
        COUNT(du.id) AS usage_count,
        COUNT(DISTINCT d.type) AS device_types_used
    FROM users u
    JOIN houses h ON u.id = h.user_id
    JOIN devices d ON h.id = d.house_id
    LEFT JOIN device_usage du ON d.id = du.device_id
    GROUP BY u.id
    ORDER BY usage_count DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []



@app.get("/analysis/device-failure-rate/", response_model=List[dict], tags=["分析功能"])
def get_device_failure_rate():
    """分析各类设备的故障率（基于安防事件）"""
    query = """
    SELECT 
        d.type AS device_type,
        COUNT(DISTINCT d.id) AS total_devices,
        COUNT(se.id) AS failure_events,
        ROUND(COUNT(se.id) * 100.0 / COUNT(DISTINCT d.id), 2) AS failure_rate_percent
    FROM devices d
    LEFT JOIN security_events se ON d.id = se.device_id
    GROUP BY d.type
    ORDER BY failure_rate_percent DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []



@app.get("/analysis/feedback-sentiment/", response_model=List[dict], tags=["分析功能"])
def get_feedback_sentiment():
    """分析用户反馈的情感分布"""
    query = """
    SELECT 
        CASE 
            WHEN rating >= 4 THEN 'Positive'
            WHEN rating = 3 THEN 'Neutral'
            ELSE 'Negative'
        END AS sentiment,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM feedback), 2) AS percentage
    FROM feedback
    GROUP BY sentiment
    ORDER BY count DESC
    """
    results = execute_query(query)
    return [dict(row) for row in results] if results else []

# 重置数据库 API
@app.post("/reset-db/", tags=["系统管理"])
def reset_database():
    """重置数据库，删除所有数据并重新创建表结构"""
    try:
        import os
        db_file = 'smart_home.db'
        if os.path.exists(db_file):
            os.remove(db_file)
            logger.info("Database file removed")

        init_db()
        return {"message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Reset DB failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-static", tags=["系统管理"])
def test_static():
    try:
        with open("static/swagger-ui-bundle.js", "r", encoding="utf-8") as f:
            content = f.read(100)  # 读取前100个字符
        return {"status": "success", "content_sample": content}
    except FileNotFoundError:
        return {"status": "error", "message": "Static file not found"}




# 健康检查端点
@app.get("/health", tags=["系统管理"])
def health_check():
    """服务健康检查"""
    try:
        # 检查数据库连接和基本查询
        execute_query("SELECT 1")
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "error": str(e)}, 500


if __name__ == "__main__":
    PORT = 9999
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', PORT))
        logger.info(f"Port {PORT} is available")
    except OSError:
        PORT = find_free_port()
        logger.warning(f"Port 9999 occupied, using free port: {PORT}")

    logger.info(f"Starting Smart Home API server on port {PORT}")

    uvicorn.run(
        "DB:app",
        host="127.0.0.1",
        port=PORT,
        reload=True,
        log_level="info"
    )