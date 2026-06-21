"""
file_server.py
MCP Server — 本地文件保存工具
 
功能：
  - write_file: 保存文案到本地 .md 文件
  - list_files: 列出已保存的文件
  - read_file:  读取已保存的文件内容
 
接入方式（claude_desktop_config.json）：
  {
    "mcpServers": {
      "file-manager": {
        "command": "python",
        "args": ["/your/path/to/file_server.py"],
        "env": {
          "OUTPUT_DIR": "/your/path/to/outputs"
        }
      }
    }
  }
"""
import os
import asyncio
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
 
# 输出目录：优先读环境变量，兜底用脚本同级的 outputs/
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
server = Server("file-manager")

# ------------------------------------------------------------------ #
# 工具列表                                                             #
# ------------------------------------------------------------------ #
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="write_file",
            description="Save generated content (e.g. social media copy) to a local markdown file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product name, used in the filename."
                    },
                    "platform": {
                        "type": "string",
                        "description": "Target platform, e.g. instagram, twitter, local.",
                        "default": "local"
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to save."
                    },
                },
                "required": ["product_name", "content"],
            },
        ),
        types.Tool(
            name="list_files",
            description="List all saved markdown files in the output directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Filter by platform prefix (optional).",
                    }
                },
            },
        ),
        types.Tool(
            name="read_file",
            description="Read the content of a saved markdown file by filename.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The filename to read (just the filename, not full path)."
                    }
                },
                "required": ["filename"],
            },
        ),
    ]
 
 
# ------------------------------------------------------------------ #
# 工具实现                                                             #
# ------------------------------------------------------------------ #
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
 
    # ── write_file ──────────────────────────────────────────────────
    if name == "write_file":
        product_name = arguments.get("product_name", "unknown")
        platform     = arguments.get("platform", "local")
        content      = arguments.get("content", "")
 
        # 与原 write_local_file 保持一致的命名规则
        safe_name = product_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{platform}_{safe_name}_{timestamp}.md"
        full_path = os.path.join(OUTPUT_DIR, filename)
 
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(f"===== 存档时间：{datetime.now()} =====\n")
                f.write(f"平台：{platform}\n产品：{product_name}\n\n")
                f.write(content)
 
            return [types.TextContent(
                type="text",
                text=f"✅ 文件已保存：{full_path}"
            )]
 
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"❌ 文件写入失败：{e}"
            )]
 
    # ── list_files ──────────────────────────────────────────────────
    elif name == "list_files":
        platform_filter = arguments.get("platform", "")
        try:
            files = [
                f for f in os.listdir(OUTPUT_DIR)
                if f.endswith(".md") and (not platform_filter or f.startswith(platform_filter))
            ]
            if not files:
                return [types.TextContent(type="text", text="📂 暂无已保存的文件")]
 
            file_list = "\n".join(f"- {f}" for f in sorted(files))
            return [types.TextContent(type="text", text=f"📂 已保存文件：\n{file_list}")]
 
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ 读取目录失败：{e}")]
 
    # ── read_file ───────────────────────────────────────────────────
    elif name == "read_file":
        filename  = arguments.get("filename", "")
        full_path = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [types.TextContent(type="text", text=content)]
 
        except FileNotFoundError:
            return [types.TextContent(type="text", text=f"❌ 文件不存在：{filename}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"❌ 读取失败：{e}")]
 
    else:
        return [types.TextContent(type="text", text=f"❌ 未知工具：{name}")]
 

# ------------------------------------------------------------------ #
# 启动                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    asyncio.run(stdio_server(server))