"""
全双工语音对话系统 v3.0 - 核心数据类型
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class DialogState(Enum):
    """对话状态"""
    IDLE = "idle"              # 空闲状态
    LISTENING = "listening"    # 聆听用户说话
    PROCESSING = "processing"  # 并行处理中
    THINKING = "thinking"      # LLM思考中
    SPEAKING = "speaking"      # AI说话中


class SemanticState(Enum):
    """语义VAD状态"""
    COMPLETE = "complete"       # 用户说完
    CONTINUING = "continuing"   # 用户继续说
    INTERRUPTED = "interrupted" # 被打断
    REJECTED = "rejected"       # 拒识/无效输入
    UNKNOWN = "unknown"         # 未知状态


class EmotionType(Enum):
    """情绪类型"""
    POSITIVE = "positive"     # 积极
    NEGATIVE = "negative"     # 消极
    NEUTRAL = "neutral"       # 中性
    ANGRY = "angry"           # 愤怒
    SAD = "sad"               # 悲伤
    SURPRISED = "surprised"   # 惊讶


@dataclass
class AudioSegment:
    """音频段"""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    duration_ms: float = 0.0
    is_speech: bool = False


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    confidence: float = 1.0
    language: str = "zh"
    is_final: bool = True
    timestamps: Optional[List[Dict]] = None  # 字级别时间戳


@dataclass
class SemanticVADResult:
    """语义VAD结果"""
    state: SemanticState
    confidence: float = 1.0
    reason: str = ""


@dataclass
class EmotionResult:
    """情绪识别结果"""
    emotion: EmotionType
    confidence: float = 1.0
    intensity: float = 0.5  # 0-1, 情绪强度
    details: Optional[Dict] = None


@dataclass
class QwenOmniResult:
    """Qwen Omni处理结果 (ASR + 语义VAD + 情绪识别)

    使用单一 Qwen-Omni 模型完成三个功能：
    1. ASR 语音转文本
    2. 语义VAD (complete/continuing/interrupted/rejected)
    3. 情绪识别 (positive/negative/neutral/angry/sad/surprised)
    """
    asr: ASRResult
    semantic_vad: SemanticVADResult
    emotion: EmotionResult = None  # 情绪识别结果
    raw_response: Optional[Dict] = None

    def __post_init__(self):
        if self.emotion is None:
            self.emotion = EmotionResult(
                emotion=EmotionType.NEUTRAL,
                confidence=0.8,
                intensity=0.5
            )


@dataclass
class LLMInput:
    """融合阶段的LLM输入"""
    text: str                              # 用户文本 (来自Qwen Omni ASR)
    text_confidence: float = 1.0           # 文本置信度
    semantic_state: SemanticState = SemanticState.COMPLETE  # 语义状态
    semantic_confidence: float = 1.0       # 语义置信度
    emotion: EmotionType = EmotionType.NEUTRAL  # 情绪类型
    emotion_confidence: float = 1.0        # 情绪置信度
    emotion_intensity: float = 0.5         # 情绪强度
    conversation_history: List[Dict] = field(default_factory=list)  # 对话历史

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "text_confidence": self.text_confidence,
            "semantic_state": self.semantic_state.value,
            "semantic_confidence": self.semantic_confidence,
            "emotion": self.emotion.value,
            "emotion_confidence": self.emotion_confidence,
            "emotion_intensity": self.emotion_intensity
        }


@dataclass
class ToolCall:
    """工具调用"""
    name: str
    arguments: Dict[str, Any]
    id: str = ""


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_call: ToolCall
    result: Any
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM响应"""
    text: str
    intent: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    final_response: str = ""  # 工具调用后的总结响应
    emotion_adapted: bool = False  # 是否进行了情绪适配
    llm_emotion: EmotionType = EmotionType.NEUTRAL  # 大模型情绪类型


@dataclass
class TTSResult:
    """TTS结果"""
    audio_data: bytes
    format: str = "mp3"
    sample_rate: int = 24000
    duration_ms: float = 0.0


@dataclass
class DialogResult:
    """完整对话结果"""
    # 输入处理
    text: str                              # 识别的文本
    text_confidence: float = 1.0           # 文本置信度
    semantic_state: SemanticState = SemanticState.COMPLETE  # 语义状态

    # 情绪
    emotion: EmotionType = EmotionType.NEUTRAL  # 情绪类型
    emotion_confidence: float = 1.0        # 情绪置信度

    # 响应
    response: str = ""                     # AI响应文本
    response_audio: Optional[bytes] = None # TTS音频

    # 状态
    is_interrupt: bool = False             # 是否是打断
    dialog_state: DialogState = DialogState.IDLE

    # 工具调用
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    # 大模型情绪
    llm_emotion: EmotionType = EmotionType.NEUTRAL  # 大模型情绪类型

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "text_confidence": self.text_confidence,
            "semantic_state": self.semantic_state.value,
            "emotion": self.emotion.value,
            "emotion_confidence": self.emotion_confidence,
            "response": self.response,
            "is_interrupt": self.is_interrupt,
            "dialog_state": self.dialog_state.value,
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls],
            "has_audio": self.response_audio is not None
        }


@dataclass
class Message:
    """对话消息"""
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    emotion: Optional[EmotionType] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # 工具名称

    def to_openai_format(self) -> Dict:
        msg = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


class InterruptState(Enum):
    """打断状态"""
    NONE = "none"                    # 无打断
    DETECTED = "detected"            # 检测到人声，等待确认
    PAUSED = "paused"                # 已暂停，缓存中
    CONFIRMED = "confirmed"          # 确认打断，开始新任务
    CANCELLED = "cancelled"          # 取消打断，恢复播放


@dataclass
class InterruptCache:
    """
    打断缓存 - 用于暂存 LLM 流式输出和 TTS 音频
    
    当检测到有效人声时：
    1. 暂停前端 LLM 回显和 TTS 播放
    2. 后端流式任务继续执行，结果缓存到此结构
    3. 如果确认打断：清空缓存，开始新任务
    4. 如果取消打断：发送缓存给前端，恢复播放
    """
    # LLM 文本缓存
    llm_chunks: List[str] = field(default_factory=list)
    llm_full_text: str = ""
    
    # TTS 音频缓存
    audio_chunks: List[bytes] = field(default_factory=list)
    
    # 缓存时间戳
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    last_update_time: float = field(default_factory=lambda: datetime.now().timestamp())
    
    # 打断状态
    state: InterruptState = InterruptState.NONE
    
    # 是否已通知前端暂停
    frontend_paused: bool = False
    
    def add_llm_chunk(self, chunk: str):
        """添加 LLM 文本块"""
        self.llm_chunks.append(chunk)
        self.llm_full_text += chunk
        self.last_update_time = datetime.now().timestamp()
    
    def add_audio_chunk(self, audio: bytes):
        """添加 TTS 音频块"""
        self.audio_chunks.append(audio)
        self.last_update_time = datetime.now().timestamp()
    
    def clear(self):
        """清空缓存"""
        self.llm_chunks.clear()
        self.llm_full_text = ""
        self.audio_chunks.clear()
        self.state = InterruptState.NONE
        self.frontend_paused = False
    
    def has_content(self) -> bool:
        """是否有缓存内容"""
        return bool(self.llm_chunks) or bool(self.audio_chunks)
    
    def get_total_audio_size(self) -> int:
        """获取总音频大小"""
        return sum(len(a) for a in self.audio_chunks)
    
    def to_dict(self) -> Dict:
        return {
            "llm_chunks_count": len(self.llm_chunks),
            "llm_full_text_length": len(self.llm_full_text),
            "audio_chunks_count": len(self.audio_chunks),
            "total_audio_size": self.get_total_audio_size(),
            "state": self.state.value,
            "frontend_paused": self.frontend_paused
        }
