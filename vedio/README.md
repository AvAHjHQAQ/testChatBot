# 全双工语音对话系统 v2.0

一个基于 Qwen Omni 的全双工语音对话系统，支持语义VAD、情绪识别、工具调用和实时打断。

## 特性

- **Qwen Omni 语义VAD + ASR**：智能判断用户是否说完
- **并行情绪识别**：与ASR并行执行，降低延迟
- **情绪融合响应**：根据用户情绪调整回复风格
- **工具调用**：支持天气查询、设置提醒、设备控制等
- **全双工交互**：支持随时打断AI说话

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"

# 或复制配置文件
cp .env.example .env
# 编辑 .env 填入API密钥
```

### 3. 启动服务

```bash
python run.py
```

### 4. 访问界面

打开浏览器访问 http://localhost:8765

## 命令行演示

如果不想使用浏览器，可以使用命令行演示：

```bash
# 交互式命令行
python demo_cli.py

# 场景测试演示
python demo_scenarios.py
```

## 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_duplex_dialog.py::TestFullDuplexScenarios -v
```

## 项目结构

```
├── config/
│   └── model_config.yaml    # 配置文件
├── src/voice_dialog/
│   ├── core/                # 核心类型和状态机
│   ├── modules/             # 功能模块
│   ├── system.py            # 核心系统
│   └── websocket_server.py  # WebSocket服务
├── web/
│   └── index.html           # Web前端
├── tests/                   # 测试代码
├── run.py                   # 启动脚本
├── demo_cli.py              # 命令行演示
└── demo_scenarios.py        # 场景演示
```

## 支持的工具

| 工具 | 功能 | 示例 |
|------|------|------|
| get_weather | 查询天气 | "北京今天天气怎么样" |
| set_reminder | 设置提醒 | "帮我设一个明天9点的提醒" |
| play_music | 播放音乐 | "播放一首轻松的歌" |
| control_device | 设备控制 | "打开客厅的灯" |
| search_web | 网络搜索 | "搜索一下Python教程" |

## 配置说明

编辑 `config/model_config.yaml` 自定义配置：

```yaml
# Qwen Omni配置
QWEN_OMNI:
  api_key: "${DASHSCOPE_API_KEY}"
  model: "qwen2.5-omni-7b"

# TTS配置
TTS:
  provider: "edge"
  voice: "zh-CN-XiaoxiaoNeural"

# 服务器配置
SERVER:
  host: "0.0.0.0"
  port: 8765
```

## 技术架构

详见 [架构设计文档.md](架构设计文档.md)

## 许可证

MIT License
