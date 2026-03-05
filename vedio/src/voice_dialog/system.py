"""
全双工语音对话系统 v3.0 - 核心系统

架构流程：
1. 声学VAD检测语音起止与打断（阈值<500ms）
2. 检测到人声开始 → 打开音频流送给ASR
3. 检测到静音 → 不立刻断，交给语义VAD判断
4. Qwen ASR 17B 流式输出文本
5. Qwen Omni Flash 语义VAD 边接收文本边判断
6. Qwen Omni Flash 情绪识别 与语义VAD并行
7. 语义完整 → 文本+情绪 → 后端LLM
8. 检测到打断 → 停止播报，开始新一轮对话
9. 支持多轮对话上下文
"""
import asyncio
from typing import Optional, Callable, List
from .core.logger import logger

from .core import (
    DialogState,
    SemanticState,
    EmotionType,
    AudioSegment,
    LLMInput,
    DialogResult,
    Message,
    DialogStateMachine,
    SemanticStateMachine,
    get_config,
)
from .modules import (
    AcousticVAD,
    StreamingVAD,
    QwenASRProcessor,
    SemanticVADProcessor,
    StreamingSemanticVAD,
    EmotionRecognizer,
    ParallelEmotionRecognizer,
    LLMTaskPlanner,
    ToolEngine,
    TTSEngine,
    StreamingTTS,
)


class VoiceDialogSystem:
    """
    全双工语音对话系统 v3.0

    核心变化：
    1. ASR独立模块（Qwen ASR 17B）流式输出
    2. 语义VAD流式判断（边接收边判断）
    3. 情绪识别与语义VAD并行
    4. 声学静音交给语义VAD决策
    5. 支持打断和多轮对话上下文
    """

    # 超时配置（毫秒）
    MAX_SILENCE_WAIT_MS = 2000  # 最大静音等待时间
    MIN_SPEECH_DURATION_MS = 300  # 最小语音时长

    def __init__(self):
        self.config = get_config()

        # 状态机
        self.dialog_state = DialogStateMachine()
        self.semantic_state = SemanticStateMachine()

        # ========== v3.0 新架构模块 ==========
        # 声学VAD（阈值<500ms）
        self.acoustic_vad = StreamingVAD()

        # 流式ASR（Qwen ASR 17B）
        self.asr_processor = QwenASRProcessor()

        # 流式语义VAD
        self.semantic_vad = StreamingSemanticVAD()

        # 并行情绪识别
        self.emotion_recognizer = ParallelEmotionRecognizer()

        # LLM和工具
        self.llm_planner = LLMTaskPlanner()
        self.tool_engine = ToolEngine()

        # TTS
        self.tts_engine = TTSEngine()
        self.streaming_tts = StreamingTTS()

        # 回调
        self._on_result_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []
        self._on_partial_asr_callbacks: List[Callable] = []
        self._on_tool_executing_callbacks: List[Callable] = []

        # ========== 流式处理状态 ==========
        self._is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._asr_text_buffer = ""
        self._streaming_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None

        # 打断控制
        self._is_interrupted = False
        self._current_tts_task: Optional[asyncio.Task] = None

        # 添加状态监听
        self.dialog_state.add_listener(self._on_state_change)

        logger.info("语音对话系统v3.0初始化完成 - 流式架构")

    async def process_audio(self, audio_chunk: bytes) -> Optional[DialogResult]:
        """
        处理音频块 - v3.0 流式架构

        流程：
        1. 检查打断
        2. 声学VAD检测
        3. 检测到人声 → 启动流式处理
        4. 检测到静音 → 交给语义VAD判断
        5. 超时保护 → 强制结束处理
        """
        import time

        # 1. 检查打断（在SPEAKING状态下检测到语音）
        if self.dialog_state.state == DialogState.SPEAKING:
            if self.acoustic_vad.check_interrupt(audio_chunk):
                return await self._handle_interrupt()

        # 2. 声学VAD处理
        vad_result = await self.acoustic_vad.process_chunk(audio_chunk)
        current_time = time.time() * 1000  # 毫秒

        # 3. 检测到人声开始
        if vad_result["event"] == "speech_active":
            if not self._is_streaming:
                # 新的语音段开始
                await self._start_streaming()
                self._is_streaming = True
                self._streaming_start_time = current_time
                self._silence_start_time = None
                await self.dialog_state.transition_to(DialogState.LISTENING, "检测到语音")
                logger.info(f"语音段开始 (时间: {current_time:.0f}ms)")

            # 更新最后语音时间
            self._last_speech_time = current_time
            self._silence_start_time = None

        # 4. 检测到静音，检查是否应该结束
        elif vad_result["event"] == "silence_detected" and self._is_streaming:
            silence_duration = vad_result["silence_duration"]

            # 记录静音开始时间
            if self._silence_start_time is None:
                self._silence_start_time = current_time

            silence_elapsed = current_time - self._silence_start_time

            logger.debug(f"检测到静音 {silence_duration:.0f}ms, 已静音 {silence_elapsed:.0f}ms")

            # 判断是否应该结束语音段
            should_finalize = False
            finalize_reason = ""

            # 条件1: 语义VAD判断完整
            if self.semantic_vad.processor.is_complete():
                should_finalize = True
                finalize_reason = "语义完整"

            # 条件2: 静音时间超过阈值且有ASR文本
            elif silence_elapsed >= self.MAX_SILENCE_WAIT_MS and self._asr_text_buffer:
                should_finalize = True
                finalize_reason = f"静音超时({silence_elapsed:.0f}ms)"

            # 条件3: 有足够文本且静音超过500ms
            elif silence_elapsed >= 500 and len(self._asr_text_buffer) >= 3:
                should_finalize = True
                finalize_reason = f"静音+文本充足"

            if should_finalize:
                logger.info(f"结束语音段: {finalize_reason}, 文本: '{self._asr_text_buffer}'")
                return await self._finalize_streaming()

        # 5. 流式处理音频（发送到ASR和情绪识别）
        if self._is_streaming:
            await self._process_audio_parallel(audio_chunk)

        return None

    async def _start_streaming(self):
        """启动流式处理"""
        logger.info("启动流式处理")

        try:
            # 启动ASR流
            asr_started = await self.asr_processor.start_stream(self._on_asr_result)
            if not asr_started:
                logger.warning("ASR流启动失败，将使用模拟模式")

            # 启动语义VAD
            await self.semantic_vad.start()

            # 启动情绪识别
            await self.emotion_recognizer.start()

            self._asr_text_buffer = ""
            self._streaming_start_time = None
            self._last_speech_time = None
            self._silence_start_time = None

            logger.info("流式处理已启动")

        except Exception as e:
            logger.error(f"启动流式处理失败: {e}")
            import traceback
            traceback.print_exc()

    async def _process_audio_parallel(self, audio_chunk: bytes):
        """
        并行处理音频

        ASR + 情绪识别并行运行
        """
        # 并行发送到ASR和情绪识别
        await asyncio.gather(
            self.asr_processor.process_chunk(audio_chunk),
            self.emotion_recognizer.process_audio(audio_chunk)
        )

    async def _on_asr_result(self, text: str, is_final: bool):
        """
        ASR流式结果回调

        边接收文本边进行语义VAD判断
        """
        self._asr_text_buffer = text

        # 通知部分ASR结果
        await self._notify_partial_asr(text)

        # 流式语义VAD判断
        if text:
            semantic_result = await self.semantic_vad.process_text(text, is_final)

            logger.debug(f"语义VAD判断: {semantic_result.state.value} (置信度: {semantic_result.confidence:.2f})")

            # 更新语义状态
            self.semantic_state.update(semantic_result.state, semantic_result.confidence)

            # 如果语义完整，准备结束
            if semantic_result.state == SemanticState.COMPLETE:
                logger.info(f"语义完整: '{text}'")
                # 不立即结束，等待声学VAD的静音确认

    async def _finalize_streaming(self) -> Optional[DialogResult]:
        """
        结束流式处理，进入融合阶段

        条件：声学静音 + 语义完整 或 超时
        """
        if not self._is_streaming:
            return None

        self._is_streaming = False

        try:
            await self.dialog_state.transition_to(DialogState.PROCESSING, "语音段结束")
        except Exception as e:
            logger.warning(f"状态转换失败: {e}")
            await self.dialog_state.force_state(DialogState.PROCESSING, "语音段结束")

        try:
            # 1. 获取最终ASR结果
            asr_result = await self.asr_processor.stop_stream()
            recognized_text = asr_result.text

            logger.info(f"ASR最终结果: '{recognized_text}'")

            # 2. 获取最终语义VAD结果
            semantic_result = await self.semantic_vad.stop()
            semantic_state = semantic_result.state
            semantic_confidence = semantic_result.confidence

            # 3. 获取情绪识别结果（句子级）
            emotion_result = await self.emotion_recognizer.finalize_sentence(recognized_text)
            emotion = emotion_result.emotion
            emotion_confidence = emotion_result.confidence
            emotion_intensity = emotion_result.intensity

            logger.info(f"处理结果: 文本='{recognized_text}', 语义={semantic_state.value}, 情绪={emotion.value}")

            # 检查有效性
            if not recognized_text or not recognized_text.strip():
                logger.warning("空输入，忽略")
                await self.dialog_state.transition_to(DialogState.IDLE, "空输入")
                return DialogResult(
                    text="",
                    semantic_state=SemanticState.REJECTED,
                    dialog_state=DialogState.IDLE
                )

            if semantic_state == SemanticState.REJECTED:
                logger.warning("拒识输入")
                await self.dialog_state.transition_to(DialogState.IDLE, "拒识输入")
                return DialogResult(
                    text="",
                    semantic_state=SemanticState.REJECTED,
                    dialog_state=DialogState.IDLE
                )

            # ========== 融合阶段：文本 + 情绪 → LLM ==========
            return await self._process_with_llm(
                recognized_text,
                asr_result.confidence,
                semantic_state,
                semantic_confidence,
                emotion,
                emotion_confidence,
                emotion_intensity
            )

        except Exception as e:
            logger.error(f"流式处理失败: {e}")
            import traceback
            traceback.print_exc()
            await self.dialog_state.force_state(DialogState.IDLE, f"错误: {e}")
            return None

    async def _process_with_llm(
        self,
        text: str,
        text_confidence: float,
        semantic_state: SemanticState,
        semantic_confidence: float,
        emotion: EmotionType,
        emotion_confidence: float,
        emotion_intensity: float
    ) -> DialogResult:
        """融合阶段：文本+情绪 → LLM处理"""
        await self.dialog_state.transition_to(DialogState.THINKING, "开始LLM处理")

        llm_input = LLMInput(
            text=text,
            text_confidence=text_confidence,
            semantic_state=semantic_state,
            semantic_confidence=semantic_confidence,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            emotion_intensity=emotion_intensity,
        )

        logger.info(f"融合输入 LLM: 文本='{text}', 情绪={emotion.value}")

        # LLM任务规划
        llm_response = await self.llm_planner.plan(llm_input)

        # 工具调用
        tool_results = []
        if llm_response.tool_calls:
            for tc in llm_response.tool_calls:
                await self._notify_tool_executing(tc.name, tc.arguments)

            logger.info(f"执行工具调用: {[tc.name for tc in llm_response.tool_calls]}")
            tool_results = await self.tool_engine.execute_batch(llm_response.tool_calls)

            final_response = await self.llm_planner.summarize_tool_results(
                llm_response, tool_results
            )
            llm_response.final_response = final_response
        else:
            llm_response.final_response = llm_response.text

        # TTS
        await self.dialog_state.transition_to(DialogState.SPEAKING, "语音合成")
        tts_result = await self.tts_engine.synthesize(llm_response.final_response)

        # 更新对话历史（支持上下文）
        self.llm_planner.add_to_history(Message(
            role="user",
            content=text,
            emotion=emotion
        ))
        self.llm_planner.add_to_history(Message(
            role="assistant",
            content=llm_response.final_response
        ))

        logger.info(f"对话完成 - 用户: '{text}' -> 助手: '{llm_response.final_response[:50]}...'")

        # 构建结果
        result = DialogResult(
            text=text,
            text_confidence=text_confidence,
            semantic_state=semantic_state,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            response=llm_response.final_response,
            response_audio=tts_result.audio_data,
            is_interrupt=False,
            dialog_state=DialogState.SPEAKING,
            tool_calls=llm_response.tool_calls,
            tool_results=tool_results,
        )

        await self._notify_result(result)
        await self.dialog_state.transition_to(DialogState.IDLE, "处理完成")

        return result

    async def process_text(self, text: str) -> DialogResult:
        """
        处理文本输入
        跳过ASR阶段，直接进入融合处理
        """
        logger.info(f"处理文本输入: '{text}'")

        if self.dialog_state.state == DialogState.IDLE:
            await self.dialog_state.transition_to(DialogState.LISTENING, "文本输入开始")

        await self.dialog_state.transition_to(DialogState.PROCESSING, "文本输入处理")

        # 文本模式下的情绪识别
        emotion_result = await self.emotion_recognizer.finalize_sentence(text)

        # 融合处理
        return await self._process_with_llm(
            text,
            1.0,  # 文本置信度
            SemanticState.COMPLETE,  # 文本模式默认完整
            1.0,
            emotion_result.emotion,
            emotion_result.confidence,
            emotion_result.intensity
        )

    async def _handle_interrupt(self) -> DialogResult:
        """处理打断"""
        self._is_interrupted = True
        self._is_streaming = False

        # 停止流式处理
        try:
            await self.asr_processor.stop_stream()
        except:
            pass

        self.semantic_vad.reset()
        self.emotion_recognizer.reset()
        self.acoustic_vad.reset()

        # 停止TTS播放
        self.streaming_tts.stop()

        # 取消当前TTS任务
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()

        # 开始新一轮对话（保持上下文）
        await self.dialog_state.force_state(DialogState.LISTENING, "用户打断")

        logger.info("已处理打断，开始新一轮对话（保持上下文）")

        return DialogResult(
            text="",
            semantic_state=SemanticState.INTERRUPTED,
            is_interrupt=True,
            dialog_state=DialogState.LISTENING
        )

    def interrupt(self):
        """手动触发打断"""
        asyncio.create_task(self._handle_interrupt())

    async def _on_state_change(self, old_state: DialogState, new_state: DialogState):
        """状态变化回调"""
        for callback in self._on_state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_state, new_state)
                else:
                    callback(old_state, new_state)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")

    async def _notify_result(self, result: DialogResult):
        """通知结果"""
        for callback in self._on_result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"结果回调错误: {e}")

    async def _notify_partial_asr(self, text: str):
        """通知部分ASR结果"""
        for callback in self._on_partial_asr_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(text)
                else:
                    callback(text)
            except Exception as e:
                logger.error(f"ASR回调错误: {e}")

    async def _notify_tool_executing(self, tool_name: str, tool_args: dict):
        """通知工具正在执行"""
        for callback in self._on_tool_executing_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tool_name, tool_args)
                else:
                    callback(tool_name, tool_args)
            except Exception as e:
                logger.error(f"工具执行回调错误: {e}")

    def on_result(self, callback: Callable):
        """注册结果回调"""
        self._on_result_callbacks.append(callback)

    def on_state_change(self, callback: Callable):
        """注册状态变化回调"""
        self._on_state_change_callbacks.append(callback)

    def on_partial_asr(self, callback: Callable):
        """注册部分ASR回调"""
        self._on_partial_asr_callbacks.append(callback)

    def on_tool_executing(self, callback: Callable):
        """注册工具执行回调"""
        self._on_tool_executing_callbacks.append(callback)

    def reset(self):
        """重置系统"""
        self.dialog_state.reset()
        self.semantic_state.reset()
        self.acoustic_vad.reset()
        self.llm_planner.clear_history()
        self._is_interrupted = False
        self._is_streaming = False
        self._asr_text_buffer = ""
        self._streaming_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None
        logger.info("系统已重置（对话历史已清空）")

    def clear_context(self):
        """仅清空上下文，保持状态"""
        self.llm_planner.clear_history()
        logger.info("对话上下文已清空")

    @property
    def current_state(self) -> DialogState:
        """当前对话状态"""
        return self.dialog_state.state

    @property
    def is_busy(self) -> bool:
        """是否忙碌"""
        return self.dialog_state.is_busy()

    @property
    def conversation_history(self) -> List[Message]:
        """获取对话历史"""
        return self.llm_planner.history