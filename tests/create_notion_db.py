import requests
import os

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID")

url = os.environ.get("NOTION_URL")

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

payload = {
    "parent": {
        "type": "page_id",
        "page_id": NOTION_PARENT_PAGE_ID
    },
    "title": [
        {"type": "text", "text": {"content": "Content Database"}}
    ],
    "properties": {
        "Name":           {"title": {}},
        "Platform":       {"select": {"options": [
            {"name": "Pinterest", "color": "red"},
            {"name": "Instagram", "color": "purple"},
            {"name": "Twitter",   "color": "blue"}
        ]}},
        "Product ID":     {"rich_text": {}},
        "Content":        {"rich_text": {}},
        "Tags":           {"multi_select": {"options": []}},
        "Publish Status": {"select": {"options": [
            {"name": "Draft",      "color": "gray"},
            {"name": "Scheduled",  "color": "yellow"},
            {"name": "Published",  "color": "green"}
        ]}},
        "Created At":     {"date": {}}
    }
}

response = requests.post(url, headers=headers, json=payload)
data = response.json()

if response.status_code == 200:
    print("✅ 创建成功！")
    print(f"Database ID: {data['id']}")
else:
    print(f"❌ 失败：{data['message']}")