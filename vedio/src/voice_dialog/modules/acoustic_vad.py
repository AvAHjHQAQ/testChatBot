"""
全双工语音对话系统 v3.0 - 声学VAD模块

核心职责（v3.0）：
1. 语音起止与打断检测
2. 阈值 < 500ms
3. 检测到人声开始 → 打开音频流，送给ASR
4. 检测到静音 → 不立刻断，交给语义VAD判断
5. 检测到打断 → 停止播报，开始新一轮对话
"""
import struct
import time
from typing import Optional, Callable, List
from ..core.logger import logger

from ..core.types import AudioSegment
from ..core.config import get_config


class WebRTCVADWrapper:
    """
    WebRTC VAD 包装器
    解决 Python 3.14 兼容性问题
    直接使用 _webrtcvad C 扩展
    """

    def __init__(self, aggressiveness: int = 3):
        self._vad = None
        self._aggressiveness = aggressiveness
        self._init_vad()

    def _init_vad(self):
        """初始化 VAD"""
        try:
            # 直接使用 _webrtcvad C 扩展
            import _webrtcvad
            self._vad = _webrtcvad.create()
            _webrtcvad.init(self._vad)
            _webrtcvad.set_mode(self._vad, self._aggressiveness)
            self._process = _webrtcvad.process
            logger.info(f"WebRTC VAD C扩展加载成功 (mode={self._aggressiveness})")
        except ImportError:
            try:
                # 尝试标准导入
                import webrtcvad
                self._vad = webrtcvad.Vad(self._aggressiveness)
                self._process = None  # 使用标准API
                logger.info(f"WebRTC VAD标准导入成功, aggressiveness={self._aggressiveness}")
            except Exception as e:
                logger.warning(f"WebRTC VAD不可用: {e}")
                self._vad = None
                self._process = None
        except Exception as e:
            logger.warning(f"WebRTC VAD初始化失败: {e}")
            self._vad = None
            self._process = None

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """检测语音"""
        if self._vad is None:
            return False
        try:
            if self._process is not None:
                # 使用 _webrtcvad.process(vad, sample_rate, buf, length)
                length = len(frame) // 2  # 16-bit samples
                return self._process(self._vad, sample_rate, frame, length)
            else:
                # 使用标准 API
                return self._vad.is_speech(frame, sample_rate)
        except Exception as e:
            logger.debug(f"VAD检测错误: {e}")
            return False

    @property
    def available(self) -> bool:
        """VAD 是否可用"""
        return self._vad is not None


class SimpleVAD:
    """
    简单音量 VAD
    基于 RMS 音量检测
    """

    def __init__(self, threshold: float = 700):  # 提高阈值以过滤更多噪声
        self.threshold = threshold

    def is_speech(self, frame: bytes, sample_rate: int = 16000) -> bool:
        """检测语音"""
        try:
            samples = struct.unpack(f'<{len(frame)//2}h', frame)
            if not samples:
                return False
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            return rms > self.threshold
        except Exception:
            return False


class AcousticVAD:
    """
    声学VAD检测器 v3.0

    核心职责：
    1. 语音起止与打断检测
    2. 阈值 < 500ms
    3. 检测到人声开始 → 打开音频流，送给ASR
    4. 检测到静音 → 不立刻断，交给语义VAD判断
    5. 检测到打断 → 停止播报，开始新一轮对话

    设计要点：
    - 检测到静音后，不立刻断开，而是通知上层交给语义VAD判断
    - 只有语义VAD确认语义完整后，才结束当前语音段
    """

    def __init__(self):
        self.config = get_config().acoustic_vad
        self._is_speech = False
        self._speech_frames: List[bytes] = []
        self._silence_frames = 0
        self._speech_callbacks: List[Callable] = []
        self._silence_callbacks: List[Callable] = []  # 静音回调（给语义VAD判断）
        self._interrupt_callbacks: List[Callable] = []  # 打断回调

        # 配置参数 - v3.1 更新：静音阈值400ms
        self.frame_duration_ms = self.config.get("frame_duration_ms", 20)  # 帧长20ms
        self.silence_threshold_ms = self.config.get("silence_threshold_ms", 400)  # 阈值400ms
        self.padding_duration_ms = self.config.get("padding_duration_ms", 150)  # padding减少
        self.sample_rate = 16000
        self.aggressiveness = self.config.get("aggressiveness", 3)

        # 打断检测增强 - 需要连续帧检测
        self._interrupt_speech_frames = 0  # 连续检测到语音的帧数
        self._interrupt_threshold_frames = 3  # 需要连续3帧(60ms)才确认打断

        # 计算帧大小
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 16-bit
        self.silence_threshold_frames = self.silence_threshold_ms // self.frame_duration_ms
        self.padding_frames = self.padding_duration_ms // self.frame_duration_ms

        # 时间追踪
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._last_silence_duration: float = 0.0

        # 初始化 VAD
        self._webrtc_vad = WebRTCVADWrapper(self.aggressiveness)
        self._simple_vad = SimpleVAD(threshold=700)  # 提高阈值过滤噪声

        if self._webrtc_vad.available:
            logger.info(f"声学VAD 使用 WebRTC VAD (阈值: {self.silence_threshold_ms}ms, 帧长: {self.frame_duration_ms}ms)")
        else:
            logger.info(f"声学VAD 使用简单音量 VAD (阈值: {self.silence_threshold_ms}ms)")

    def process_frame(self, audio_frame: bytes) -> Optional[AudioSegment]:
        """
        处理音频帧

        v3.0 逻辑：
        - 检测到人声开始 → 返回 None，触发语音开始事件
        - 检测到静音 → 返回 None，触发静音事件（交给语义VAD判断）
        - 只有语义VAD确认后，才会调用 finalize_segment 获取完整语音段

        Returns:
            检测到的完整语音段，或None
        """
        is_speech = self._detect_speech(audio_frame)

        if is_speech:
            self._silence_frames = 0
            self._speech_frames.append(audio_frame)
            self._silence_start_time = None
            self._last_silence_duration = 0.0

            if not self._is_speech:
                self._is_speech = True
                self._speech_start_time = time.time()
                logger.debug("声学VAD: 检测到语音开始")
                # 触发语音开始回调
                self._notify_speech_start()

        else:
            if self._is_speech:
                self._silence_frames += 1
                self._speech_frames.append(audio_frame)  # 保留静音部分作为padding

                # 记录静音开始时间
                if self._silence_start_time is None:
                    self._silence_start_time = time.time()

                # 计算当前静音时长
                self._last_silence_duration = (time.time() - self._silence_start_time) * 1000

                # 检查是否达到静音阈值
                if self._silence_frames >= self.silence_threshold_frames:
                    logger.debug(f"声学VAD: 检测到静音 (时长: {self._last_silence_duration:.0f}ms)，等待语义VAD判断")
                    # 触发静音回调（不结束语音段，交给语义VAD判断）
                    self._notify_silence_detected(self._last_silence_duration)

        return None

    def _detect_speech(self, audio_frame: bytes) -> bool:
        """检测单帧是否包含语音"""
        # 优先使用 WebRTC VAD
        if self._webrtc_vad.available:
            try:
                if len(audio_frame) == self.frame_size:
                    return self._webrtc_vad.is_speech(audio_frame, self.sample_rate)
            except Exception:
                pass

        # 回退到简单 VAD
        return self._simple_vad.is_speech(audio_frame)

    def finalize_segment(self) -> Optional[AudioSegment]:
        """
        结束当前语音段（由语义VAD调用）

        当语义VAD确认语义完整后，调用此方法获取完整语音段

        Returns:
            完整的语音段
        """
        if not self._speech_frames:
            return None

        audio_data = b''.join(self._speech_frames)
        duration_ms = len(audio_data) / self.sample_rate / 2 * 1000

        logger.debug(f"声学VAD: 语音段结束, 长度={len(audio_data)} bytes, 时长={duration_ms:.0f}ms")

        segment = AudioSegment(
            data=audio_data,
            sample_rate=self.sample_rate,
            is_speech=True,
            duration_ms=duration_ms
        )

        # 重置状态
        self._is_speech = False
        self._speech_frames.clear()
        self._silence_frames = 0
        self._speech_start_time = None
        self._silence_start_time = None
        self._last_silence_duration = 0.0

        return segment

    def check_interrupt(self, audio_frame: bytes) -> bool:
        """
        检查是否是打断信号

        v3.2 更新：需要连续检测到多帧语音才确认打断，减少误触发

        Returns:
            是否检测到打断
        """
        is_speech = self._detect_speech(audio_frame)

        if is_speech:
            self._interrupt_speech_frames += 1
            # 需要连续检测到足够的帧数才确认打断
            if self._interrupt_speech_frames >= self._interrupt_threshold_frames:
                logger.info(f"声学VAD: 确认打断信号 (连续{self._interrupt_speech_frames}帧)")
                self._interrupt_speech_frames = 0  # 重置计数
                self._notify_interrupt()
                return True
        else:
            # 非语音帧，重置计数
            self._interrupt_speech_frames = 0

        return False

    def get_silence_duration(self) -> float:
        """
        获取当前静音时长（毫秒）

        Returns:
            静音时长（ms）
        """
        return self._last_silence_duration

    def reset(self):
        """重置状态"""
        self._is_speech = False
        self._speech_frames.clear()
        self._silence_frames = 0
        self._speech_start_time = None
        self._silence_start_time = None
        self._last_silence_duration = 0.0
        self._interrupt_speech_frames = 0  # 重置打断检测计数

    def add_speech_callback(self, callback: Callable):
        """添加语音开始回调"""
        self._speech_callbacks.append(callback)

    def add_silence_callback(self, callback: Callable):
        """添加静音检测回调"""
        self._silence_callbacks.append(callback)

    def add_interrupt_callback(self, callback: Callable):
        """添加打断回调"""
        self._interrupt_callbacks.append(callback)

    def _notify_speech_start(self):
        """通知语音开始"""
        for callback in self._speech_callbacks:
            try:
                callback("speech_start")
            except Exception as e:
                logger.error(f"语音开始回调错误: {e}")

    def _notify_silence_detected(self, duration_ms: float):
        """通知检测到静音"""
        for callback in self._silence_callbacks:
            try:
                callback("silence_detected", duration_ms)
            except Exception as e:
                logger.error(f"静音回调错误: {e}")

    def _notify_interrupt(self):
        """通知打断"""
        for callback in self._interrupt_callbacks:
            try:
                callback("interrupt")
            except Exception as e:
                logger.error(f"打断回调错误: {e}")

    @property
    def is_speech_active(self) -> bool:
        """当前是否有语音活动"""
        return self._is_speech

    @property
    def current_audio_buffer(self) -> bytes:
        """获取当前音频缓冲区"""
        return b''.join(self._speech_frames)


class StreamingVAD:
    """
    流式VAD处理器 v3.0
    支持异步处理音频流
    """

    def __init__(self):
        self.acoustic_vad = AcousticVAD()
        self._buffer = bytearray()
        self._frame_size = self.acoustic_vad.frame_size

    async def process_chunk(self, audio_chunk: bytes) -> dict:
        """
        处理音频块

        Returns:
            处理结果字典，包含：
            - event: "speech_start" / "silence_detected" / "none"
            - silence_duration: 静音时长（ms）
            - is_interrupt: 是否是打断
        """
        self._buffer.extend(audio_chunk)
        result = {
            "event": "none",
            "silence_duration": 0.0,
            "is_interrupt": False
        }

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            self._buffer = self._buffer[self._frame_size:]

            # 处理帧
            self.acoustic_vad.process_frame(frame)

            # 检查状态
            if self.acoustic_vad._is_speech and self.acoustic_vad._last_silence_duration > 0:
                result["event"] = "silence_detected"
                result["silence_duration"] = self.acoustic_vad.get_silence_duration()
            elif self.acoustic_vad._is_speech:
                result["event"] = "speech_active"

        return result

    def check_interrupt(self, audio_chunk: bytes) -> bool:
        """
        检查是否是打断

        Returns:
            是否检测到打断
        """
        # 处理缓冲区中的帧
        self._buffer.extend(audio_chunk)

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            self._buffer = self._buffer[self._frame_size:]

            if self.acoustic_vad.check_interrupt(frame):
                return True

        return False

    def finalize_segment(self) -> Optional[AudioSegment]:
        """结束当前语音段"""
        return self.acoustic_vad.finalize_segment()

    def get_silence_duration(self) -> float:
        """获取当前静音时长"""
        return self.acoustic_vad.get_silence_duration()

    def reset(self):
        """重置"""
        self._buffer.clear()
        self.acoustic_vad.reset()

    @property
    def is_speech_active(self) -> bool:
        """当前是否有语音活动"""
        return self.acoustic_vad.is_speech_active