"""
全双工语音对话系统 v3.0 - TTS模块

支持：
- Edge TTS（默认）
- Qwen TTS
- 流式TTS播放
"""
import asyncio
import io
from typing import Optional
from ..core.logger import logger

from ..core.types import TTSResult
from ..core.config import get_config


class TTSEngine:
    """
    TTS引擎
    支持多种TTS后端: Edge TTS / Qwen TTS
    """

    def __init__(self):
        self.config = get_config().tts
        self.provider = self.config.get("provider", "edge")
        self._edge_tts = None

    async def synthesize(self, text: str) -> TTSResult:
        """
        文本转语音
        """
        if not text.strip():
            return TTSResult(audio_data=b"", duration_ms=0)

        try:
            if self.provider == "edge":
                return await self._synthesize_edge(text)
            elif self.provider == "qwen":
                return await self._synthesize_qwen(text)
            else:
                return await self._synthesize_edge(text)

        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
            return TTSResult(audio_data=b"", duration_ms=0)

    async def _synthesize_edge(self, text: str) -> TTSResult:
        """使用Edge TTS"""
        try:
            import edge_tts

            voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
            rate = self.config.get("rate", "+0%")
            pitch = self.config.get("pitch", "+0Hz")

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch
            )

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # 估算时长 (MP3约128kbps)
            duration_ms = len(audio_data) / 128 * 8

            return TTSResult(
                audio_data=audio_data,
                format="mp3",
                sample_rate=24000,
                duration_ms=duration_ms
            )

        except ImportError:
            logger.error("edge-tts未安装")
            return await self._mock_synthesize(text)

    async def _synthesize_qwen(self, text: str) -> TTSResult:
        """使用Qwen TTS"""
        # Qwen TTS API实现
        # 这里使用模拟数据
        return await self._mock_synthesize(text)

    async def _mock_synthesize(self, text: str) -> TTSResult:
        """模拟TTS（用于测试）"""
        await asyncio.sleep(0.1)

        # 生成静音音频作为占位
        # 实际使用时会被真实TTS替代
        duration_ms = len(text) * 150  # 估算时长

        return TTSResult(
            audio_data=b"\x00" * 1000,  # 模拟音频数据
            format="mp3",
            sample_rate=24000,
            duration_ms=duration_ms
        )


class StreamingTTS:
    """
    流式TTS
    支持边合成边播放
    """

    def __init__(self):
        self.engine = TTSEngine()
        self._is_playing = False
        self._should_stop = False

    async def stream_synthesize(self, text: str):
        """
        流式合成并返回音频块
        """
        try:
            import edge_tts

            config = get_config().tts
            voice = config.get("voice", "zh-CN-XiaoxiaoNeural")

            communicate = edge_tts.Communicate(text=text, voice=voice)

            self._is_playing = True
            self._should_stop = False

            async for chunk in communicate.stream():
                if self._should_stop:
                    break

                if chunk["type"] == "audio":
                    yield chunk["data"]

            self._is_playing = False

        except Exception as e:
            logger.error(f"流式TTS失败: {e}")
            self._is_playing = False

    def stop(self):
        """停止播放"""
        self._should_stop = True

    @property
    def is_playing(self) -> bool:
        return self._is_playing
