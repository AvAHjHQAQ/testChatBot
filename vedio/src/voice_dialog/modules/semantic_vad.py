"""
全双工语音对话系统 v3.0 - 语义VAD流式判断模块

职责：
- 流式文本实时检测
- 边接收ASR文本边判断语义完整性
- 使用 Qwen Omni Flash 模型

关键设计：
- 语义VAD负责语义完整性判断（"说完"、"拒识"、"是不是真打断"）
- 满足声学静音阈值或自己判断出语义完整了，再给后端LLM模型推理
"""
import asyncio
import json
from typing import Optional, List, Dict
from ..core.logger import logger

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ..core.types import SemanticState, SemanticVADResult
from ..core.config import get_config


class SemanticVADProcessor:
    """
    语义VAD流式处理器

    使用 Qwen3-Omni-Flash 模型进行流式语义判断
    边接收ASR文本边判断语义完整性
    """

    MODEL_NAME = "Qwen3-Omni-Flash"

    # 语义判断提示词
    SEMANTIC_PROMPT = """分析用户的输入文本，判断用户的表达状态。

语义状态说明：
- complete: 用户完整表达了一个意图或问题，可以开始处理
- continuing: 用户还在说，表达不完整，需要继续等待
- interrupted: 用户被打断或中途停止
- rejected: 无法识别或无效输入（如噪音、无意义声音）

判断要点：
1. 是否有完整的意图或问题？
2. 句子是否完整（主谓宾齐全）？
3. 是否有明显的结束标记（问号、句号）？
4. 是否还有后续内容的可能性？

请以JSON格式返回：
{
    "semantic_state": "complete/continuing/interrupted/rejected",
    "confidence": 0.95,
    "reason": "判断理由"
}"""

    def __init__(self):
        self.config = get_config().semantic_vad if hasattr(get_config(), 'semantic_vad') else {}
        self.api_key = get_config().qwen_omni.get("api_key", "")
        self.model = self.config.get("model", self.MODEL_NAME)

        # 流式状态
        self._text_buffer = ""
        self._last_state = SemanticState.CONTINUING
        self._last_confidence = 0.5
        self._judgment_history: List[Dict] = []

        # 配置
        self.min_text_length = self.config.get("streaming", {}).get("min_text_length", 2)
        self.max_wait_ms = self.config.get("streaming", {}).get("max_wait_ms", 5000)

        self._init_client()

    def _init_client(self):
        """初始化 API 客户端"""
        if not HAS_OPENAI:
            logger.warning("openai 库未安装，将使用规则判断")
            return

        if self.api_key:
            logger.info(f"语义VAD 客户端初始化成功 (模型: {self.model})")
        else:
            logger.warning("未配置 API 密钥，将使用规则判断")

    async def judge(self, text: str, is_final: bool = False) -> SemanticVADResult:
        """
        判断文本的语义完整性

        Args:
            text: ASR流式输出的文本
            is_final: 是否是最终文本

        Returns:
            语义VAD判断结果
        """
        # 更新缓冲区
        self._text_buffer = text

        # 文本太短，继续等待（但不等待太久）
        if len(text.strip()) < 1:
            self._last_state = SemanticState.CONTINUING
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.7,
                reason="文本为空，继续等待"
            )

        # 如果是最终结果，使用更宽松的判断
        if is_final:
            result = self._judge_with_rules(text)
            # 如果仍然不是完整，但有内容，强制完整
            if result.state != SemanticState.COMPLETE and len(text.strip()) >= 2:
                result = SemanticVADResult(
                    state=SemanticState.COMPLETE,
                    confidence=0.7,
                    reason="用户说话结束，判断为完整"
                )
        else:
            # 使用模型判断
            if HAS_OPENAI and self.api_key:
                result = await self._judge_with_model(text)
            else:
                result = self._judge_with_rules(text)

        # 更新状态
        self._last_state = result.state
        self._last_confidence = result.confidence

        # 记录判断历史
        self._judgment_history.append({
            "text": text,
            "state": result.state.value,
            "confidence": result.confidence,
            "reason": result.reason
        })

        return result

    async def _judge_with_model(self, text: str) -> SemanticVADResult:
        """使用 Qwen Omni Flash 模型进行语义判断"""
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 使用 Qwen Omni Flash 模型
            response = await client.chat.completions.create(
                model=self.MODEL_NAME,  # Qwen3-Omni-Flash
                messages=[
                    {
                        "role": "system",
                        "content": self.SEMANTIC_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"判断以下文本的语义状态：\n{text}"
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=100
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)

            # 解析状态
            state_str = result_json.get("semantic_state", "continuing")
            if state_str not in ["complete", "continuing", "interrupted", "rejected"]:
                state_str = "continuing"

            return SemanticVADResult(
                state=SemanticState(state_str),
                confidence=result_json.get("confidence", 0.8),
                reason=result_json.get("reason", "")
            )

        except Exception as e:
            logger.error(f"语义VAD模型判断失败: {e}")
            return self._judge_with_rules(text)

    def _judge_with_rules(self, text: str) -> SemanticVADResult:
        """基于规则的语义判断 - 优化版，更容易判断完整"""
        text = text.strip()

        # 空文本或太短
        if not text or len(text) < 1:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.6,
                reason="文本为空或太短"
            )

        # 忽略噪声词 - 这些词不应该单独触发判断
        noise_words = ["嗯", "啊", "呃", "那个", "就是", "这个", "然后", "所以", "但是", "其实", "额", "唔"]
        clean_text = text
        for noise in noise_words:
            clean_text = clean_text.replace(noise, "")
        clean_text = clean_text.strip()

        # 如果只剩下噪声，继续等待
        if not clean_text or len(clean_text) < 2:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.7,
                reason="只是噪声词，继续等待"
            )

        # 结束标点（强信号）
        end_punctuation = text[-1] if text else ""
        has_end_mark = end_punctuation in ["。", "！", "？", "!", "?", "."]

        # 问句判断（问号或疑问词）
        is_question = "？" in text or "?" in text or any(
            text.endswith(q) or q in text for q in ["吗", "呢", "吧", "么"]
        )

        # 完整意图关键词（用户说这些词时通常已经完成表达）
        complete_indicators = [
            # 动作类
            "帮我", "请", "我要", "我想", "查一下", "播放", "设置", "打开", "关闭",
            "查询", "告诉我", "怎么样", "找", "搜", "看", "听", "读", "写",
            # 信息类
            "是什么", "怎么", "如何", "为什么", "哪", "谁", "几", "多少",
            # 结束语
            "好的", "好的", "可以", "行", "谢谢", "感谢", "再见", "拜拜"
        ]
        has_intent = any(ind in text for kw in complete_indicators for ind in [kw] if kw in text)

        # 长度判断 - 如果够长，更可能完整
        text_length = len(clean_text)

        # ========== 判断逻辑 ==========

        # 1. 有结束标点 + 有实际内容 → 大概率完整
        if has_end_mark and len(clean_text) >= 2:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.9,
                reason="有结束标点且有实质内容"
            )

        # 2. 问句 → 大概率完整
        if is_question and len(clean_text) >= 2:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.85,
                reason="是问句"
            )

        # 3. 有意图关键词 + 长度足够 → 较高概率完整
        if has_intent and text_length >= 3:
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.8,
                reason="包含意图关键词"
            )

        # 4. 纯文本长度足够（>= 4个有效字符）→ 可能完整
        # 降低长度要求，让用户说完后更容易被判断为完整
        if text_length >= 4:
            # 检查是否像是在说话中途
            mid_sentence_hints = ["而且", "并且", "然后", "还有", "另外", "还有呢"]
            is_mid_sentence = any(hint in text for hint in mid_sentence_hints)
            if not is_mid_sentence:
                return SemanticVADResult(
                    state=SemanticState.COMPLETE,
                    confidence=0.7,
                    reason="文本长度足够，判断为完整"
                )

        # 5. 长度较短但有结束标记特征
        if text_length >= 2 and (has_end_mark or is_question):
            return SemanticVADResult(
                state=SemanticState.COMPLETE,
                confidence=0.75,
                reason="短句但符合完整特征"
            )

        # 6. 默认继续等待
        return SemanticVADResult(
            state=SemanticState.CONTINUING,
            confidence=0.5,
            reason="继续等待更多输入"
        )

    def is_complete(self) -> bool:
        """
        检查当前语义状态是否完整

        Returns:
            是否语义完整
        """
        return self._last_state == SemanticState.COMPLETE

    @property
    def current_state(self) -> SemanticState:
        """当前语义状态"""
        return self._last_state

    @property
    def current_text(self) -> str:
        """当前缓冲的文本"""
        return self._text_buffer

    def reset(self):
        """重置状态"""
        self._text_buffer = ""
        self._last_state = SemanticState.CONTINUING
        self._last_confidence = 0.5
        self._judgment_history.clear()
        logger.debug("语义VAD 状态已重置")

    def get_judgment_history(self) -> List[Dict]:
        """获取判断历史"""
        return self._judgment_history.copy()


class StreamingSemanticVAD:
    """
    流式语义VAD处理器
    与ASR流式输出配合使用
    """

    def __init__(self):
        self.processor = SemanticVADProcessor()
        self._is_active = False
        self._result_queue = asyncio.Queue()

    async def start(self):
        """启动流式处理"""
        self._is_active = True
        self.processor.reset()
        logger.info("流式语义VAD 已启动")

    async def process_text(self, text: str, is_final: bool = False) -> SemanticVADResult:
        """
        处理ASR流式输出的文本

        Args:
            text: ASR输出的文本
            is_final: 是否是最终文本

        Returns:
            语义VAD判断结果
        """
        if not self._is_active:
            return SemanticVADResult(
                state=SemanticState.CONTINUING,
                confidence=0.5,
                reason="未启动"
            )

        result = await self.processor.judge(text, is_final)

        # 如果语义完整，放入队列
        if result.state == SemanticState.COMPLETE:
            await self._result_queue.put(result)

        return result

    async def wait_for_complete(self, timeout: float = 5.0) -> Optional[SemanticVADResult]:
        """
        等待语义完整

        Args:
            timeout: 超时时间（秒）

        Returns:
            语义完整的结果，或超时返回None
        """
        try:
            return await asyncio.wait_for(
                self._result_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def stop(self) -> SemanticVADResult:
        """停止流式处理并返回最终结果"""
        self._is_active = False

        # 如果缓冲区有内容，进行最终判断
        if self.processor.current_text:
            # 进行最终判断
            final_result = await self.processor.judge(
                self.processor.current_text,
                is_final=True
            )

            # 如果仍然是 CONTINUING，检查文本长度
            # 如果有足够内容，强制设置为 COMPLETE
            if final_result.state == SemanticState.CONTINUING:
                clean_text = self.processor.current_text.strip()
                # 去除噪声词
                noise_words = ["嗯", "啊", "呃", "那个", "就是", "这个", "然后"]
                for noise in noise_words:
                    clean_text = clean_text.replace(noise, "")
                clean_text = clean_text.strip()

                # 如果有3个以上有效字符，认为完整
                if len(clean_text) >= 3:
                    final_result = SemanticVADResult(
                        state=SemanticState.COMPLETE,
                        confidence=0.7,
                        reason="用户停止说话，根据内容判断为完整"
                    )
                elif len(clean_text) >= 1:
                    # 有一些内容，可能完整
                    final_result = SemanticVADResult(
                        state=SemanticState.COMPLETE,
                        confidence=0.6,
                        reason="用户停止说话，内容较短但判断为完整"
                    )
                else:
                    # 只有噪声
                    final_result = SemanticVADResult(
                        state=SemanticState.REJECTED,
                        confidence=0.7,
                        reason="只有噪声，忽略"
                    )

            return final_result

        return SemanticVADResult(
            state=self.processor.current_state,
            confidence=self.processor._last_confidence,
            reason="流式处理结束"
        )

    @property
    def is_active(self) -> bool:
        """是否正在处理"""
        return self._is_active

    def reset(self):
        """重置"""
        self.processor.reset()
        self._result_queue = asyncio.Queue()