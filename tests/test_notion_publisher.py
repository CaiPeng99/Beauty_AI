"""
test_notion_publisher.py
独立测试脚本 — 不依赖项目其他模块，单独运行即可

运行方式：
    python test_notion_publisher.py
"""

import os
import sys

# ── 1. 检查依赖 ────────────────────────────────────────────────────────────
try:
    from notion_client import Client
    from notion_client.errors import APIResponseError
except ImportError:
    print("❌ 缺少依赖，请先运行：pip install notion-client python-dotenv")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv 未安装，将直接读取系统环境变量")

# ── 2. 读取凭证 ────────────────────────────────────────────────────────────
NOTION_TOKEN       = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

def check_env():
    print("\n🔍 Step 1: 检查环境变量")
    ok = True
    if not NOTION_TOKEN:
        print("  ❌ NOTION_TOKEN 未设置")
        ok = False
    else:
        print(f"  ✅ NOTION_TOKEN: {NOTION_TOKEN[:12]}...")

    if not NOTION_DATABASE_ID:
        print("  ❌ NOTION_DATABASE_ID 未设置")
        ok = False
    else:
        print(f"  ✅ NOTION_DATABASE_ID: {NOTION_DATABASE_ID[:8]}...")

    if not ok:
        print("\n👉 请在 .env 文件里添加：")
        print("   NOTION_TOKEN=secret_xxxxxxxxxx")
        print("   NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        sys.exit(1)

# ── 3. 测试 Token 是否有效 ─────────────────────────────────────────────────
def test_auth(client: Client):
    print("\n🔍 Step 2: 验证 Token 有效性")
    try:
        # 用 users/me 测试认证
        me = client.users.me()
        name = me.get("name") or me.get("id", "unknown")
        print(f"  ✅ Token 有效，Integration 名称: {name}")
        return True
    except APIResponseError as e:
        print(f"  ❌ Token 无效: {e.message}")
        print("  👉 请去 notion.so/my-integrations 重新复制 Internal Integration Secret")
        return False

# ── 4. 测试 Database 是否可访问 ────────────────────────────────────────────
def test_database(client: Client):
    print("\n🔍 Step 3: 验证 Database 可访问性")
    try:
        db = client.databases.retrieve(database_id=NOTION_DATABASE_ID)
        title_parts = db.get("title", [])
        db_name = title_parts[0]["plain_text"] if title_parts else "(无标题)"
        print(f"  ✅ Database 可访问，名称: {db_name}")

        # 列出所有 property 名，方便排查列名是否匹配
        props = list(db.get("properties", {}).keys())
        print(f"  📋 现有列: {props}")
        return True
    except APIResponseError as e:
        print(f"  ❌ Database 无法访问: {e.message}")
        if "Could not find database" in str(e):
            print("  👉 请确认：")
            print("     1. Database ID 是否正确（URL 里 32 位字符串）")
            print("     2. Database 是否已 Share 给你的 Integration")
            print("        → 打开 Database → 右上角 ··· → Connections → 选择你的 Integration")
        return False

# ── 5. 测试写入一条测试数据 ────────────────────────────────────────────────
def test_create_page(client: Client):
    print("\n🔍 Step 4: 测试写入一条 Page")

    # 模拟 generate_content 输出的文案
    mock_content = (
        "✨ Just tried the NARS Natural Radiant Longwear Foundation "
        "and I'm obsessed! 92% of users recommend it — full coverage "
        "that lasts all day without feeling heavy. Perfect for dry skin! 🌟"
    )
    mock_tags    = ["NARSBeauty", "FoundationReview", "MakeupOfTheDay"]
    mock_product = "P12345"

    from datetime import datetime, timezone

    try:
        response = client.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Name": {
                    "title": [{"text": {"content": "✨ Just tried the NARS Natural Radiant Longwear Foundation…"}}]
                },
                "Platform": {
                    "select": {"name": "notion"}
                },
                "Product ID": {
                    "rich_text": [{"text": {"content": mock_product}}]
                },
                "Content": {
                    "rich_text": [{"text": {"content": mock_content[:200]}}]
                },
                "Tags": {
                    "multi_select": [{"name": t} for t in mock_tags]
                },
                "Publish Status": {
                    "select": {"name": "draft"}
                },
                "Created At": {
                    "date": {"start": datetime.now(timezone.utc).isoformat()}
                },
            },
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": mock_content}}]
                    },
                },
                {"object": "block", "type": "divider", "divider": {}},
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{"type": "text", "text": {
                            "content": "  ".join(f"#{t}" for t in mock_tags)
                        }}],
                        "icon": {"emoji": "🏷️"},
                        "color": "pink_background",
                    },
                },
            ],
        )

        page_id  = response["id"]
        page_url = response.get("url", f"https://notion.so/{page_id.replace('-', '')}")
        print(f"  ✅ Page 创建成功！")
        print(f"  🔗 链接: {page_url}")
        return page_id

    except APIResponseError as e:
        print(f"  ❌ 写入失败: {e.message}")

        # 常见错误提示
        if "is not a property" in str(e) or "property" in str(e).lower():
            print("\n  👉 列名不匹配！请对照以下列名检查你的 Database：")
            print("     Name          → Title 类型")
            print("     Platform      → Select 类型")
            print("     Product ID    → Rich Text 类型")
            print("     Content       → Rich Text 类型")
            print("     Tags          → Multi-select 类型")
            print("     Publish Status→ Select 类型")
            print("     Created At    → Date 类型")
        return None

# ── 6. 测试 update_status ─────────────────────────────────────────────────
def test_update_status(client: Client, page_id: str):
    print("\n🔍 Step 5: 测试更新 Page 状态")
    try:
        client.pages.update(
            page_id=page_id,
            properties={"Publish Status": {"select": {"name": "published"}}},
        )
        print("  ✅ 状态更新成功: draft → published")
    except APIResponseError as e:
        print(f"  ❌ 状态更新失败: {e.message}")

# ── 主流程 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Notion Publisher 测试脚本")
    print("=" * 55)

    check_env()

    client = Client(auth=NOTION_TOKEN)

    if not test_auth(client):
        sys.exit(1)

    if not test_database(client):
        sys.exit(1)

    page_id = test_create_page(client)
    if page_id:
        test_update_status(client, page_id)

    print("\n" + "=" * 55)
    if page_id:
        print("  🎉 全部通过！去 Notion 里确认一下新建的 Page 吧")
    else:
        print("  ❌ 有步骤未通过，根据上面的提示修复后重新运行")
    print("=" * 55)