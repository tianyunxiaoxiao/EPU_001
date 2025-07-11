"""
长期低频EPU (长期低频经济政策不确定性) 提示词配置
用于识别和量化长期内低频变化的经济政策不确定性内容
"""

LONG_FREQ_EPU_PROMPT = """
你是一个专业的经济政策分析师。请分析以下新闻文本，评估其中包含的"长期低频经济政策不确定性"程度。

长期低频EPU定义：指影响时间跨度长(数年)、变化频率低但影响深远的经济政策相关不确定性。

评估标准：
1. 时间跨度：政策影响周期长，涉及长期经济结构
2. 频率：政策变化相对稳定，调整频率低
3. 深度影响：对经济基本面和长期预期的影响
4. 政策类型：结构性改革、制度变革、长期规划

请给出0-100的分数，其中：
0-20：几乎无长期低频EPU内容
21-40：轻微长期低频EPU内容
41-60：中等长期低频EPU内容
61-80：较强长期低频EPU内容
81-100：极强长期低频EPU内容

Few-shot示例：

输入：政府启动新一轮国企改革，改革方案仍在制定中，预期将持续数年。
输出：80
解释：结构性改革，时间跨度长，影响深远，低频但重要。

输入：养老保险制度改革进入讨论阶段，具体方案和时间表待定。
输出：75
解释：制度性改革，长期影响，变化频率低但意义重大。

输入：央行今日开展逆回购操作。
输出：10
解释：短期货币政策操作，非长期低频政策。

现在请分析以下新闻文本：
{news_text}

请直接输出分数(0-100的整数)：
"""

def get_long_freq_epu_prompt(news_text):
    """获取长期低频EPU分析提示词"""
    return LONG_FREQ_EPU_PROMPT.format(news_text=news_text)
