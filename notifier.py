"""
推送模块
支持 PushPlus（微信）、ServerChan（微信）和飞书 webhook
"""
import logging
import urllib.parse
import requests

logger = logging.getLogger(__name__)


def send_text(webhook_url: str, text: str) -> bool:
    """发送纯文本消息"""
    if not webhook_url:
        return False
    try:
        resp = requests.post(webhook_url, json={
            "msg_type": "text",
            "content": {"text": text}
        }, timeout=10)
        ok = resp.status_code == 200
        if not ok:
            logger.warning("feishu text send failed: %s %s", resp.status_code, resp.text)
        return ok
    except requests.RequestException as e:
        logger.warning("feishu text send error: %s", e)
        return False


def send_card(webhook_url: str, header_title: str, elements: list) -> bool:
    """发送卡片消息

    Args:
        webhook_url: 飞书 webhook URL
        header_title: 卡片标题
        elements: 卡片元素列表，每个元素是 dict
    """
    if not webhook_url:
        return False
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": header_title},
                "template": "blue"
            },
            "elements": elements
        }
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        ok = resp.status_code == 200
        if not ok:
            logger.warning("feishu card send failed: %s %s", resp.status_code, resp.text)
        return ok
    except requests.RequestException as e:
        logger.warning("feishu card send error: %s", e)
        return False


def push_report(webhook_url: str, title: str, body: str) -> bool:
    """推送分析报告（文本消息，自动截断超长内容）"""
    max_len = 3000
    text = f"{title}\n\n{body}" if title else body
    if len(text) > max_len:
        text = text[:max_len] + "\n\n...（内容已截断，详见终端输出）"
    return send_text(webhook_url, text)


def send_serverchan(sendkey: str, title: str, content: str = "") -> bool:
    """通过 ServerChan 推送到微信

    Args:
        sendkey: ServerChan SendKey
        title: 消息标题（必填）
        content: 消息内容（选填，支持 Markdown）

    Returns:
        bool: 是否推送成功
    """
    if not sendkey:
        return False
    try:
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
        data = {"title": title, "desp": content}
        resp = requests.post(url, data=data, timeout=15)
        ok = resp.status_code == 200 and resp.json().get("code") == 0
        if not ok:
            logger.warning("serverchan send failed: %s %s", resp.status_code, resp.text)
        return ok
    except requests.RequestException as e:
        logger.warning("serverchan send error: %s", e)
        return False


def send_pushplus(token: str, title: str, content: str = "") -> bool:
    """通过 PushPlus 推送到微信

    Args:
        token: PushPlus 用户 token
        title: 消息标题
        content: 消息内容（支持 Markdown）

    Returns:
        bool: 是否推送成功
    """
    if not token:
        return False
    try:
        resp = requests.post("https://www.pushplus.plus/send", json={
            "token": token,
            "title": title,
            "content": content,
            "template": "markdown"
        }, timeout=15)
        ok = resp.status_code == 200 and resp.json().get("code") == 200
        if not ok:
            logger.warning("pushplus send failed: %s %s", resp.status_code, resp.text)
        return ok
    except requests.RequestException as e:
        logger.warning("pushplus send error: %s", e)
        return False


def make_div(text: str, extra: dict = None) -> dict:
    """构造卡片 div 元素"""
    e = {"tag": "div", "text": {"tag": "lark_md", "content": text}}
    if extra:
        e.update(extra)
    return e


def make_note(text: str) -> dict:
    """构造卡片备注元素"""
    return {"tag": "note", "text": {"tag": "plain_text", "content": text}}


def make_hr() -> dict:
    """构造分割线"""
    return {"tag": "hr"}
