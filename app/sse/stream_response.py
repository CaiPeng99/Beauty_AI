from typing import AsyncGenerator
import asyncio

async def sse_stream_generator(step_messages: list) -> AsyncGenerator[str, None]:
    """
    SSE 流式生成器
    逐行返回流程步骤，模拟实时输出
    """
    for msg in step_messages:
        yield f"data: {msg}\n\n"
        await asyncio.sleep(0.8)  # 模拟延迟，展示流式效果