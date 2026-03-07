"""
全双工语音对话系统 v3.0 - 情绪识别模块

v3.0 核心变化：
- 使用 Qwen Omni Flash 模型
- 与语义VAD并行运行
- 基于音频流输入判断情绪
- 每次只判断完整一句话的情绪
- 下一轮对话重新开始判断
"""
import asyncio
import json
import time
from typing import Optional, Dict, List
from ..core.logger import logger

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False

from ..core.types import AudioSegment, EmotionResult, EmotionType
from ..core.config import get_config


# 扩展的情绪关键词（用于快速匹配）
EMOTION_KEYWORDS = {
    "positive": [
        "太好了", "谢谢", "感谢", "棒", "太棒", "很好", "开心", "高兴", "满意",
        "不错", "好的", "太赞", "厉害", "完美", "喜欢", "爱了", "优秀",
        "非常好", "好极了", "赞", "感谢你", "谢谢你", "多谢"
    ],
    "negative": [
        "糟糕", "不行", "失败", "难过", "差", "不好", "失望", "太差", "很差",
        "不好用", "不满意", "体验差", "质量差", "太差了", "很失望", "糟糕透了"
    ],
    "angry": [
        "生气", "烦死了", "烦人", "讨厌", "可恶", "混蛋", "什么破", "太慢了",
        "气死", "火大", "恼火", "愤怒", "不爽", "烦", "别烦我", "够了",
        "不想听", "别说了", "闭嘴"
    ],
    "sad": [
        "伤心", "难过", "哭", "悲伤", "心痛", "心疼", "郁闷", "不开心",
        "好难过", "好伤心", "想哭", "难受", "失落", "沮丧"
    ],
    "surprised": [
        "哇", "天哪", "不会吧", "真的吗", "没想到", "太神奇", "不可思议",
        "震惊", "惊讶", "意外", "居然", "竟然"
    ]
}


class EmotionRecognizer:
    """
    情绪识别器 v3.0

    关键设计：
    - 使用 Qwen Omni Flash 模型
    - 基于音频流输入判断情绪
    - 与语义VAD并行运行
    - 每次只判断完整一句话的情绪
    - 下一轮对话重新开始判断
    """

    MODEL_NAME = "Qwen3-Omni-Flash"

    EMOTION_PROMPT = """分析用户的语音输入，判断用户的情绪状态，返回JSON格式：
{
    "emotion": "positive/negative/neutral/angry/sad/surprised",
    "confidence": 0.95,
    "intensity": 0.8,
    "reason": "判断理由"
}

情绪说明：
- positive: 开心、满意、感谢等积极情绪
- negative: 不满、失望、沮丧等消极情绪
- neutral: 中性、平淡
- angry: 愤怒、生气、不耐烦
- sad: 悲伤、难过
- surprised: 惊讶、意外"""

    def __init__(self):
        self.config = get_config().emotion if hasattr(get_config(), 'emotion') else {}
        self.api_key = get_config().qwen_omni.get("api_key", "")
        self.model = self.config.get("model", self.MODEL_NAME)
        self.keywords = EMOTION_KEYWORDS
        self.client: Optional[AsyncOpenAI] = None

        # 并行处理状态
        self._is_processing = False
        self._audio_buffer = bytearray()
        self._sentence_complete = False
        self._current_emotion: Optional[EmotionResult] = None

        # 句子级情绪判断（每次只判断完整一句话）
        self._sentence_start_time: Optional[float] = None
        self._last_emotion_time: Optional[float] = None

        self._init_client()

    def _init_client(self):
        """初始化API客户端"""
        if not HAS_OPENAI:
            return

        if self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            logger.info(f"情绪识别器初始化成功 (模型: {self.model}, 并行模式)")

    async def process_audio_chunk(self, audio_chunk: bytes) -> Optional[EmotionResult]:
        """
        处理音频块 - 并行模式

        与语义VAD并行运行，基于音频流输入判断情绪

        Args:
            audio_chunk: 音频数据

        Returns:
            情绪识别结果（句子完成时返回）
        """
        # 累积音频数据
        self._audio_buffer.extend(audio_chunk)

        # 记录句子开始时间
        if self._sentence_start_time is None:
            self._sentence_start_time = time.time()

        return None  # 累积模式，在句子完成时返回结果

    async def finalize_sentence(self, text: Optional[str] = None) -> EmotionResult:
        """
        句子完成时进行情绪判断

        每次只判断完整一句话的情绪

        Args:
            text: 对应的ASR文本（可选，用于辅助判断）

        Returns:
            情绪识别结果
        """
        if not self._audio_buffer and not text:
            return EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.5,
                intensity=0.5
            )

        self._is_processing = True
        self._sentence_complete = True

        try:
            # 优先使用文本进行情绪判断
            if text:
                result = await self._recognize_from_text_or_audio(text, self._audio_buffer)
            else:
                # 纯音频判断
                result = await self._recognize_from_audio(bytes(self._audio_buffer))

            self._current_emotion = result
            self._last_emotion_time = time.time()

            logger.info(f"句子情绪识别完成: {result.emotion.value} (置信度: {result.confidence:.2f})")

            return result

        finally:
            self._is_processing = False
            # 重置状态，准备下一轮对话
            self.reset_sentence()

    async def _recognize_from_text_or_audio(self, text: str, audio_data: bytes) -> EmotionResult:
        """
        基于文本和音频进行情绪识别

        优先使用文本关键词快速匹配，必要时调用模型
        """
        # 1. 先尝试关键词快速匹配
        text_result = self._recognize_from_text(text)
        if text_result.confidence > 0.8:
            return text_result

        # 2. 使用模型进行更精确的判断
        if self.client and text:
            try:
                return await self._recognize_with_llm(text)
            except Exception as e:
                logger.warning(f"模型情绪识别失败: {e}")

        # 3. 返回文本识别结果
        return text_result

    async def _recognize_from_audio(self, audio_data: bytes) -> EmotionResult:
        """基于音频进行情绪识别"""
        # 分析音频能量和特征
        energy_result = self._analyze_audio_energy(audio_data)

        # 如果没有模型，返回基于能量的结果
        if not self.client:
            return energy_result

        # TODO: 调用 Qwen Omni Flash 的音频情绪识别API
        # 目前返回基于能量的结果
        return energy_result

    def _analyze_audio_energy(self, audio_data: bytes) -> EmotionResult:
        """分析音频能量特征判断情绪"""
        import struct
        try:
            samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
            if not samples:
                return EmotionResult(emotion=EmotionType.NEUTRAL, confidence=0.5, intensity=0.5)

            # 计算RMS和动态范围
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            max_val = max(abs(s) for s in samples)
            dynamic_range = max_val / (rms + 1)

            # 基于能量特征判断
            if rms > 10000:
                # 高能量 = 可能是激动/愤怒
                return EmotionResult(
                    emotion=EmotionType.ANGRY,
                    confidence=0.6,
                    intensity=0.8,
                    details={"rms": rms, "reason": "高能量"}
                )
            elif rms > 5000:
                # 中等能量 = 积极/惊讶
                return EmotionResult(
                    emotion=EmotionType.POSITIVE,
                    confidence=0.5,
                    intensity=0.6,
                    details={"rms": rms}
                )
            else:
                # 低能量 = 中性
                return EmotionResult(
                    emotion=EmotionType.NEUTRAL,
                    confidence=0.6,
                    intensity=0.4,
                    details={"rms": rms}
                )

        except Exception as e:
            logger.error(f"音频能量分析错误: {e}")
            return EmotionResult(emotion=EmotionType.NEUTRAL, confidence=0.5, intensity=0.5)

    def _recognize_from_text(self, text: str) -> EmotionResult:
        """基于关键词的文本情绪识别"""
        text_lower = text.lower()

        # 检查各情绪类型的关键词
        scores = {}
        matched = {}
        for emotion_name, keywords in self.keywords.items():
            if not isinstance(keywords, list):
                continue
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                scores[emotion_name] = len(matches)
                matched[emotion_name] = matches

        if not scores:
            return EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.6,
                intensity=0.3
            )

        # 找到最高分的情绪
        best_emotion = max(scores, key=scores.get)
        max_score = scores[best_emotion]

        try:
            emotion_type = EmotionType(best_emotion)
        except ValueError:
            emotion_type = EmotionType.NEUTRAL

        return EmotionResult(
            emotion=emotion_type,
            confidence=min(0.5 + max_score * 0.15, 0.95),
            intensity=min(0.4 + max_score * 0.2, 1.0),
            details={"matched_keywords": matched}
        )

    async def _recognize_with_llm(self, text: str) -> EmotionResult:
        """使用LLM进行情绪识别"""
        messages = [
            {"role": "system", "content": self.EMOTION_PROMPT},
            {"role": "user", "content": f"分析以下文本的情绪：\n{text}"}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return EmotionResult(
                emotion=EmotionType(result.get("emotion", "neutral")),
                confidence=result.get("confidence", 0.8),
                intensity=result.get("intensity", 0.5),
                details={"reason": result.get("reason", "")}
            )

        except Exception as e:
            logger.error(f"LLM情绪识别错误: {e}")
            raise

    def reset_sentence(self):
        """
        重置句子级状态

        下一轮对话重新开始判断
        """
        self._audio_buffer.clear()
        self._sentence_complete = False
        self._sentence_start_time = None
        self._is_processing = False
        logger.debug("情绪识别器句子状态已重置")

    def reset(self):
        """完全重置状态"""
        self.reset_sentence()
        self._current_emotion = None
        self._last_emotion_time = None
        logger.info("情绪识别器已完全重置")

    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        return self._is_processing

    @property
    def current_emotion(self) -> Optional[EmotionResult]:
        """当前情绪结果"""
        return self._current_emotion

    @property
    def sentence_duration(self) -> float:
        """当前句子时长（秒）"""
        if self._sentence_start_time:
            return time.time() - self._sentence_start_time
        return 0.0


class ParallelEmotionRecognizer:
    """
    并行情绪识别器

    与语义VAD并行运行，每次只判断完整一句话的情绪
    """

    def __init__(self):
        self.recognizer = EmotionRecognizer()
        self._emotion_queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """启动并行识别"""
        self._running = True
        self.recognizer.reset()
        logger.info("并行情绪识别器已启动")

    async def process_audio(self, audio_chunk: bytes):
        """
        处理音频数据

        Args:
            audio_chunk: 音频数据
        """
        if self._running:
            await self.recognizer.process_audio_chunk(audio_chunk)

    async def finalize_sentence(self, text: Optional[str] = None) -> EmotionResult:
        """
        句子完成时进行情绪判断

        Args:
            text: ASR文本

        Returns:
            情绪识别结果
        """
        result = await self.recognizer.finalize_sentence(text)
        await self._emotion_queue.put(result)
        return result

    async def get_emotion(self, timeout: float = 2.0) -> Optional[EmotionResult]:
        """
        获取情绪识别结果

        Args:
            timeout: 超时时间

        Returns:
            情绪识别结果
        """
        try:
            return await asyncio.wait_for(
                self._emotion_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def stop(self) -> Optional[EmotionResult]:
        """停止并获取最终结果"""
        self._running = False
        return self.recognizer.current_emotion

    def reset(self):
        """重置状态"""
        self.recognizer.reset()
        self._emotion_queue = asyncio.Queue()


# 便捷函数：用于快速文本情绪识别
def recognize_emotion_from_text(text: str) -> EmotionResult:
    """
    从文本快速识别情绪（同步方法）
    用于文本输入模式
    """
    recognizer = EmotionRecognizer()
    return recognizer._recognize_from_text(text)