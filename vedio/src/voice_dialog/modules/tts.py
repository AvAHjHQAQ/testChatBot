"""
全双工语音对话系统 v3.2 - TTS模块

支持：
- Edge TTS（默认）
- Qwen TTS
- 流式TTS播放
- LLM流式输出实时转语音
"""
import asyncio
import io
import re
from typing import Optional, Callable, AsyncIterator, List
from ..core.logger import logger

from ..core.types import TTSResult
from ..core.config import get_config


def clean_text_for_tts(text: str) -> str:
    """
    清理文本中的Markdown格式符号和表情符号，使其适合TTS播报

    处理内容：
    - Markdown格式符号（粗体、斜体、删除线、链接等）
    - 表情符号
    - 多余的空白字符
    """
    if not text:
        return text

    original_text = text

    # 1. 处理代码块（先处理多行的）
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', '').strip(), text)
    text = re.sub(r'`([^`]+?)`', r'\1', text)  # 行内代码

    # 2. 处理链接 [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 3. 处理图片 ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # 4. 处理Markdown格式符号
    # 粗体 **text** 和 __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)

    # 斜体 *text* 和 _text_（注意避免匹配下划线变量名）
    text = re.sub(r'(?<![a-zA-Z0-9])\*([^*]+?)\*(?![*])', r'\1', text)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+?)_(?![a-zA-Z0-9_])', r'\1', text)

    # 删除线 ~~text~~ 和 --text--
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'--(.+?)--', r'\1', text)

    # 5. 处理标题 # ## ### 等
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 6. 处理引用 > text
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # 7. 处理列表符号
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # 无序列表
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 有序列表

    # 8. 处理分隔线 *** --- ___
    text = re.sub(r'^(\*{3,}|-{3,}|_{3,})\s*$', '', text, flags=re.MULTILINE)

    # 9. 处理HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 10. 移除表情符号（使用Unicode范围）
    # 注意：范围必须精确，避免误删中文字符
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和象形文字
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F700-\U0001F77F"  # 炼金术符号
        "\U0001F780-\U0001F7FF"  # 几何图形扩展
        "\U0001F800-\U0001F8FF"  # 补充箭头-C
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
        "\U0001FA00-\U0001FA6F"  # 国际象棋符号
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A
        "\U00002702-\U000027B0"  # 装饰符号
        "\U0001F200-\U0001F251"  # 包围字符补充（修复：原来是\U000024C2，会误删中文）
        "\U0001F004"             # 麻将牌
        "\U0001F0CF"             # 扑克牌
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # 11. 清理多余的空白字符
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = text.strip()

    # 如果清理后为空，返回原文（防止过度清理）
    if not text and original_text:
        return original_text

    return text


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
        # 清理文本中的格式符号和表情
        cleaned_text = clean_text_for_tts(text)

        if not cleaned_text.strip():
            return TTSResult(audio_data=b"", duration_ms=0)

        try:
            if self.provider == "edge":
                return await self._synthesize_edge(cleaned_text)
            elif self.provider == "qwen":
                return await self._synthesize_qwen(cleaned_text)
            else:
                return await self._synthesize_edge(cleaned_text)

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
        # 清理文本中的格式符号和表情
        cleaned_text = clean_text_for_tts(text)

        if not cleaned_text.strip():
            return

        try:
            import edge_tts

            config = get_config().tts
            voice = config.get("voice", "zh-CN-XiaoxiaoNeural")

            communicate = edge_tts.Communicate(text=cleaned_text, voice=voice)

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


class StreamingTTSProcessor:
    """
    流式TTS处理器 v3.3

    核心功能：
    - 接收LLM流式输出的文本块
    - 按句子分段转换
    - 每个句子完成后立即发送音频（流畅播报）
    - 使用队列确保先进先出顺序

    分段策略：
    - 遇到句子结束符（。！？等）时，立即转换这个句子并发送
    - 确保每个音频块是完整句子，播放更自然
    """

    # 句子结束符
    SENTENCE_ENDINGS = ['。', '！', '？', '；', '\n']

    # 配置参数
    MIN_CHUNK_SIZE = 5       # 最小分段长度（字符）
    MAX_CHUNK_SIZE = 80      # 最大分段长度（字符）

    def __init__(self, on_audio_chunk: Optional[Callable[[bytes], None]] = None):
        """
        初始化流式TTS处理器

        Args:
            on_audio_chunk: 音频块回调函数
        """
        self.config = get_config().tts
        self.on_audio_chunk = on_audio_chunk

        # 文本缓冲区
        self._text_buffer = ""

        # 状态
        self._should_stop = False

        # 异步锁 - 确保顺序处理
        self._lock = asyncio.Lock()

        # 统计
        self._total_text = ""
        self._audio_chunks: List[bytes] = []
        self._sentence_count = 0  # 句子计数，用于调试

        logger.debug("StreamingTTSProcessor 初始化完成")

    async def add_text(self, text: str) -> bool:
        """
        添加LLM输出的文本块（线程安全，按顺序处理）

        Args:
            text: 文本块

        Returns:
            是否触发了TTS转换
        """
        if self._should_stop or not text:
            return False

        # 使用锁确保顺序处理
        async with self._lock:
            if self._should_stop:
                return False

            self._text_buffer += text
            self._total_text += text

            # 检查是否需要转换
            should_synthesize = self._should_synthesize()

            if should_synthesize:
                await self._synthesize_buffer_locked()

        return should_synthesize

    def _should_synthesize(self) -> bool:
        """判断是否需要进行TTS转换 - 按句子分段"""
        if not self._text_buffer:
            return False

        # 检查是否有句子结束符
        for ending in self.SENTENCE_ENDINGS:
            if ending in self._text_buffer:
                return True

        # 超过最大长度也要转换
        if len(self._text_buffer) >= self.MAX_CHUNK_SIZE:
            return True

        return False

    def _split_text(self) -> tuple:
        """
        分割文本，返回可以转换的部分和剩余部分

        Returns:
            (待转换文本, 剩余文本)
        """
        if not self._text_buffer:
            return "", ""

        # 查找句子结束位置
        best_pos = -1
        for ending in self.SENTENCE_ENDINGS:
            pos = self._text_buffer.find(ending)
            if pos != -1:
                if best_pos == -1 or pos < best_pos:
                    best_pos = pos

        if best_pos != -1:
            # 找到句子结束符，在结束符后分割
            split_pos = best_pos + 1
            to_synthesize = self._text_buffer[:split_pos]
            remaining = self._text_buffer[split_pos:]
            return to_synthesize, remaining

        # 没有找到句子结束符，检查长度
        if len(self._text_buffer) >= self.MAX_CHUNK_SIZE:
            # 按最大长度分割（尽量在空格或标点处分割）
            split_pos = self.MAX_CHUNK_SIZE

            # 尝试找到更好的分割点
            for i in range(self.MAX_CHUNK_SIZE - 1, self.MIN_CHUNK_SIZE, -1):
                if i < len(self._text_buffer) and self._text_buffer[i] in ['，', ',', ' ', '、']:
                    split_pos = i + 1
                    break

            to_synthesize = self._text_buffer[:split_pos]
            remaining = self._text_buffer[split_pos:]
            return to_synthesize, remaining

        return "", self._text_buffer

    async def _synthesize_buffer_locked(self):
        """转换缓冲区中的文本（已持有锁）"""
        to_synthesize, remaining = self._split_text()

        if not to_synthesize:
            return

        self._text_buffer = remaining

        # 清理文本
        cleaned_text = clean_text_for_tts(to_synthesize)

        if not cleaned_text.strip():
            return

        self._sentence_count += 1
        sentence_num = self._sentence_count

        try:
            # 使用Edge TTS进行转换
            import edge_tts

            voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
            rate = self.config.get("rate", "+0%")

            communicate = edge_tts.Communicate(
                text=cleaned_text,
                voice=voice,
                rate=rate
            )

            # 收集完整音频数据，最后一次性发送
            audio_data = b""
            async for chunk in communicate.stream():
                if self._should_stop:
                    break

                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # 只有在音频完整后才发送
            if audio_data and self.on_audio_chunk and not self._should_stop:
                self._audio_chunks.append(audio_data)
                await self._call_callback(audio_data)

            logger.debug(f"[TTS] 句子#{sentence_num} 转换完成: '{cleaned_text[:30]}...' -> {len(audio_data)} bytes")

        except Exception as e:
            logger.error(f"[TTS] 句子#{sentence_num} 转换失败: {e}")

    async def _call_callback(self, audio_data: bytes):
        """调用音频回调"""
        if self.on_audio_chunk:
            try:
                if asyncio.iscoroutinefunction(self.on_audio_chunk):
                    await self.on_audio_chunk(audio_data)
                else:
                    self.on_audio_chunk(audio_data)
            except Exception as e:
                logger.error(f"音频回调错误: {e}")

    async def flush(self):
        """
        刷新缓冲区，转换剩余文本（使用锁确保顺序）
        """
        async with self._lock:
            if self._should_stop:
                return

            if self._text_buffer and len(self._text_buffer) >= self.MIN_CHUNK_SIZE:
                # 保存剩余文本
                to_synthesize = self._text_buffer
                self._text_buffer = ""

                # 清理并转换
                cleaned_text = clean_text_for_tts(to_synthesize)

                if cleaned_text.strip():
                    self._sentence_count += 1
                    sentence_num = self._sentence_count

                    try:
                        import edge_tts

                        voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
                        rate = self.config.get("rate", "+0%")

                        communicate = edge_tts.Communicate(
                            text=cleaned_text,
                            voice=voice,
                            rate=rate
                        )

                        # 收集完整音频数据
                        audio_data = b""
                        async for chunk in communicate.stream():
                            if self._should_stop:
                                break

                            if chunk["type"] == "audio":
                                audio_data += chunk["data"]

                        # 发送音频
                        if audio_data and self.on_audio_chunk and not self._should_stop:
                            self._audio_chunks.append(audio_data)
                            await self._call_callback(audio_data)

                        logger.debug(f"[TTS] 句子#{sentence_num}(flush) 转换完成: '{cleaned_text[:30]}...' -> {len(audio_data)} bytes")

                    except Exception as e:
                        logger.error(f"[TTS] flush转换失败: {e}")

    async def finalize(self) -> bytes:
        """
        完成处理，返回所有音频数据

        Returns:
            完整的音频数据
        """
        # 刷新剩余缓冲区（内部已使用锁）
        await self.flush()

        # 合并所有音频块
        all_audio = b"".join(self._audio_chunks)

        logger.info(f"[TTS] 处理完成: 总文本 {len(self._total_text)} 字符, 句子数 {self._sentence_count}, 音频 {len(all_audio)} bytes")

        return all_audio

    def stop(self):
        """停止处理"""
        self._should_stop = True

    def reset(self):
        """重置状态"""
        self._text_buffer = ""
        self._total_text = ""
        self._audio_chunks = []
        self._should_stop = False
        self._sentence_count = 0

    @property
    def is_processing(self) -> bool:
        return self._lock.locked()

    @property
    def total_text(self) -> str:
        return self._total_text
