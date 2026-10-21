"""整理模型寫出、會直接顯示在網站上的文字。

提示詞已經列出台灣用語，但 Flash-Lite 仍會寫出下面這幾個中國大陸用語。
依序替換：「最優」要排在「優化」前面，「最優化」才會變成「最佳化」而不是「最最佳化」。
"""

import re

TERM_FIXES = {
    "最優": "最佳",
    "優化": "最佳化",
    "數據": "資料",
    "政權切換": "體制轉換",
    "政權轉換": "體制轉換",
    "宏觀": "總體",       # 宏觀經濟 → 總體經濟
    "對沖": "避險",       # 對沖基金 → 避險基金
}


def fix_terms(text: str) -> str:
    for wrong, right in TERM_FIXES.items():
        text = text.replace(wrong, right)
    # 模型有時兩種說法並用，例如「總體宏觀」，替換後會變成「總體總體」
    for right in set(TERM_FIXES.values()):
        while right * 2 in text:
            text = text.replace(right * 2, right)
    return text


# 模型偶爾把提示詞的字數要求抄進輸出，例如「……軌跡，40 字以內。」。
# 只刪結尾的，句中的「N 字以內」可能是原意
_LENGTH_NOTE = re.compile(r"[，,；;]?\s*[（(]?\s*\d+\s*字(?:以內|以内|內|内|為限)\s*[）)]?(?=[。.]?$)")


def clean_text(text: str) -> str:
    """換成台灣用語，並刪掉抄進來的字數要求。"""
    return _LENGTH_NOTE.sub("", fix_terms(text.strip())).strip()
