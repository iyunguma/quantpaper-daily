"""把模型常寫出的中國大陸用語換成台灣用語。

提示詞已經列出台灣用語，但 Flash-Lite 仍會寫出下面這幾個，而且會直接顯示在網站上。
依序替換：「最優」要排在「優化」前面，「最優化」才會變成「最佳化」而不是「最最佳化」。
"""

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
