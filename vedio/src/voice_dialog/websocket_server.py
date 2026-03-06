"""
全双工语音对话系统 v3.2 - WebSocket服务器
支持全双工语音交互

v3.2 特性：
- 流式ASR + 流式语义VAD
- 并行情绪识别
- 打断支持
- 实时时延监控
- LLM流式输出
"""
import asyncio
import json
import base64
from typing import Dict, Set, Callable
from .core.logger import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from .system import VoiceDialogSystem
from .core import DialogState, DialogResult, latency_tracker


app = FastAPI(title="全双工语音对话系统 v3.2")


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.dialog_systems: Dict[str, VoiceDialogSystem] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # 为每个连接创建独立的对话系统
        system = VoiceDialogSystem()
        self.dialog_systems[client_id] = system

        # 注册回调
        system.on_result(lambda r: asyncio.create_task(
            self.send_result(client_id, r)
        ))
        system.on_state_change(lambda old, new: asyncio.create_task(
            self.send_state_change(client_id, old, new)
        ))
        system.on_partial_asr(lambda text: asyncio.create_task(
            self.send_partial_asr(client_id, text)
        ))
        system.on_tool_executing(lambda tool_name, tool_args: asyncio.create_task(
            self.send_tool_executing(client_id, tool_name, tool_args)
        ))
        system.on_latency_update(lambda data: asyncio.create_task(
            self.send_latency_update(client_id, data)
        ))
        # 注册LLM流式输出回调
        system.on_llm_chunk(lambda chunk: asyncio.create_task(
            self.send_llm_chunk(client_id, chunk)
        ))
        # 注册TTS音频块回调（实时播报）
        system.on_audio_chunk(lambda audio: asyncio.create_task(
            self.send_audio_chunk(client_id, audio)
        ))
        # 注册清空音频回调（新问题开始时）
        system.on_clear_audio(lambda: asyncio.create_task(
            self.send_clear_audio(client_id)
        ))

        logger.info(f"客户端连接: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.dialog_systems:
            del self.dialog_systems[client_id]
        logger.info(f"客户端断开: {client_id}")

    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")

    async def send_result(self, client_id: str, result: DialogResult):
        """发送对话结果 - 只发送LLM总结后的响应，不发送原始工具结果"""
        data = {
            "type": "result",
            "data": {
                "text": result.text,
                "text_confidence": result.text_confidence,
                "semantic_state": result.semantic_state.value,
                "emotion": result.emotion.value,
                "emotion_confidence": result.emotion_confidence,
                "response": result.response,  # 这是LLM总结后的响应
                "is_interrupt": result.is_interrupt,
                "dialog_state": result.dialog_state.value,
                # 只显示工具名称，不返回执行结果
                "tool_calls": [{"name": tc.name} for tc in result.tool_calls],
                "has_audio": result.response_audio is not None
            }
        }

        # 如果有音频，编码为base64
        if result.response_audio:
            data["data"]["audio"] = base64.b64encode(result.response_audio).decode()

        await self.send_json(client_id, data)

    async def send_llm_chunk(self, client_id: str, chunk: str):
        """发送LLM流式输出块"""
        await self.send_json(client_id, {
            "type": "llm_chunk",
            "data": {"text": chunk}
        })

    async def send_audio_chunk(self, client_id: str, audio_data: bytes):
        """发送TTS音频块（实时播报）"""
        await self.send_json(client_id, {
            "type": "audio_chunk",
            "data": {"audio": base64.b64encode(audio_data).decode()}
        })

    async def send_clear_audio(self, client_id: str):
        """发送清空音频队列消息"""
        await self.send_json(client_id, {
            "type": "clear_audio"
        })
        logger.info(f"[音频流] 已通知前端清空音频队列: {client_id}")

    async def send_tool_executing(self, client_id: str, tool_name: str, tool_args: dict):
        """发送工具执行状态"""
        await self.send_json(client_id, {
            "type": "tool_executing",
            "data": {
                "tool_name": tool_name,
                "tool_args": tool_args
            }
        })

    async def send_state_change(self, client_id: str, old_state: DialogState, new_state: DialogState):
        """发送状态变化"""
        await self.send_json(client_id, {
            "type": "state_change",
            "data": {
                "old_state": old_state.value,
                "new_state": new_state.value
            }
        })

    async def send_partial_asr(self, client_id: str, text: str):
        """发送部分ASR结果"""
        await self.send_json(client_id, {
            "type": "partial_asr",
            "data": {"text": text}
        })

    async def send_latency_update(self, client_id: str, data):
        """发送时延更新"""
        if data is None:
            return
        await self.send_json(client_id, {
            "type": "latency_update",
            "data": data.to_dict() if hasattr(data, 'to_dict') else data
        })

    def get_system(self, client_id: str) -> VoiceDialogSystem:
        return self.dialog_systems.get(client_id)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive()

            if "text" in data:
                # JSON消息
                message = json.loads(data["text"])
                await handle_message(client_id, message)

            elif "bytes" in data:
                # 音频数据 (PCM 16kHz 16bit mono)
                audio_chunk = data["bytes"]
                await handle_audio(client_id, audio_chunk)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(client_id)


async def handle_message(client_id: str, message: dict):
    """处理JSON消息"""
    msg_type = message.get("type", "")
    system = manager.get_system(client_id)

    if not system:
        return

    if msg_type == "text":
        # 文本输入
        text = message.get("text", "")
        if text:
            result = await system.process_text(text)
            # 结果会通过回调发送

    elif msg_type == "interrupt":
        # 打断
        await system.interrupt()

    elif msg_type == "reset":
        # 重置
        system.reset()
        await manager.send_json(client_id, {"type": "reset", "data": {"success": True}})

    elif msg_type == "ping":
        # 心跳
        await manager.send_json(client_id, {"type": "pong"})


async def handle_audio(client_id: str, audio_chunk: bytes):
    """处理音频数据 - 支持全双工实时语音"""
    system = manager.get_system(client_id)

    if not system:
        return

    # 直接处理音频，支持实时语音交互
    result = await system.process_audio(audio_chunk)
    # 结果会通过回调发送


@app.get("/")
async def get_index():
    """返回主页"""
    return FileResponse("web/index.html")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "version": "3.0"}


@app.get("/latency/history")
async def get_latency_history(limit: int = 10):
    """获取时延历史记录"""
    history = latency_tracker.get_history(limit)
    return {
        "history": [h.to_dict() for h in history],
        "total": len(history)
    }


@app.get("/latency/stats")
async def get_latency_stats():
    """获取时延统计信息"""
    return latency_tracker.get_stats()


@app.get("/latency/current")
async def get_current_latency():
    """获取当前时延数据"""
    current = latency_tracker.get_current()
    if current:
        return current.to_dict()
    return {"status": "no_active_sentence"}


@app.get("/monitor")
async def get_monitor():
    """返回时延监控页面"""
    return FileResponse("web/latency_monitor.html")


@app.get("/interrupt-test")
async def get_interrupt_test():
    """返回打断测试页面"""
    return FileResponse("web/interrupt_test.html")


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """启动服务器"""
    from .core.config import get_config
    config = get_config()

    server_config = config.server
    host = server_config.get("host", host)
    port = server_config.get("port", port)

    logger.info(f"启动WebSocket服务器: ws://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()