from openai import OpenAI
from app.config import ZhiPu_API_KEY, LLM_MODEL

class ZhiPuLLMAdapter:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model  = model

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=20.0,
        )
        return resp.choices[0].message.content.strip()

_raw_client = OpenAI(
    api_key=ZhiPu_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

llm_adapter = ZhiPuLLMAdapter(_raw_client, LLM_MODEL)