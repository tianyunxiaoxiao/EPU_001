"""
提示词格式化模块
为不同类型的EPU分析生成格式化的提示词
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.prompts.good_epu_prompt import get_good_epu_prompt
from config.prompts.bad_epu_prompt import get_bad_epu_prompt
from config.prompts.short_freq_epu_prompt import get_short_freq_epu_prompt
from config.prompts.long_freq_epu_prompt import get_long_freq_epu_prompt
from config.prompts.total_epu_prompt import get_total_epu_prompt
from config.config import Config

class PromptFormatter:
    """提示词格式化器"""
    
    def __init__(self):
        self.epu_prompt_functions = {
            'good_epu': get_good_epu_prompt,
            'bad_epu': get_bad_epu_prompt,
            'short_freq_epu': get_short_freq_epu_prompt,
            'long_freq_epu': get_long_freq_epu_prompt,
            'total_epu': get_total_epu_prompt
        }
    
    def format_news_content(self, news_content_list, max_length=2000):
        """格式化新闻内容为单个文本"""
        if not news_content_list:
            return ""
        
        # 合并所有新闻内容
        combined_text = "\n\n".join(news_content_list)
        
        # 如果文本太长，截取前面部分
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        return combined_text
    
    def get_epu_prompt(self, epu_type, news_content):
        """获取指定类型的EPU分析提示词"""
        if epu_type not in self.epu_prompt_functions:
            raise ValueError(f"不支持的EPU类型: {epu_type}")
        
        # 格式化新闻内容
        if isinstance(news_content, list):
            formatted_content = self.format_news_content(news_content)
        else:
            formatted_content = str(news_content)
        
        # 获取对应的提示词函数并生成提示词
        prompt_function = self.epu_prompt_functions[epu_type]
        return prompt_function(formatted_content)
    
    def get_all_epu_prompts(self, news_content):
        """获取所有类型的EPU分析提示词"""
        prompts = {}
        
        for epu_type in Config.EPU_TYPES:
            try:
                prompts[epu_type] = self.get_epu_prompt(epu_type, news_content)
            except Exception as e:
                print(f"生成{epu_type}提示词失败: {str(e)}")
                prompts[epu_type] = None
        
        return prompts
    
    def prepare_api_requests(self, news_data_list):
        """为API请求准备数据"""
        requests_data = []
        
        for news_data in news_data_list:
            date = news_data.get('date', 'unknown')
            content = news_data.get('content', [])
            
            if not content:
                continue
            
            # 为每种EPU类型生成请求数据
            for epu_type in Config.EPU_TYPES:
                try:
                    prompt = self.get_epu_prompt(epu_type, content)
                    
                    request_data = {
                        'date': date,
                        'epu_type': epu_type,
                        'epu_type_name': Config.EPU_TYPE_NAMES.get(epu_type, epu_type),
                        'prompt': prompt,
                        'original_content': content,
                        'content_count': len(content)
                    }
                    
                    requests_data.append(request_data)
                    
                except Exception as e:
                    print(f"准备{date}-{epu_type}请求数据失败: {str(e)}")
                    continue
        
        return requests_data
    
    def validate_prompt(self, prompt):
        """验证提示词格式"""
        if not prompt or not isinstance(prompt, str):
            return False
        
        # 检查提示词长度
        if len(prompt) < 100:
            return False
        
        # 检查是否包含必要元素
        required_elements = ['分析', '评估', '分数']
        for element in required_elements:
            if element not in prompt:
                return False
        
        return True
    
    def get_prompt_statistics(self, prompts_dict):
        """获取提示词统计信息"""
        stats = {}
        
        for epu_type, prompt in prompts_dict.items():
            if prompt:
                stats[epu_type] = {
                    'length': len(prompt),
                    'valid': self.validate_prompt(prompt),
                    'word_count': len(prompt.split())
                }
            else:
                stats[epu_type] = {
                    'length': 0,
                    'valid': False,
                    'word_count': 0
                }
        
        return stats

def main():
    """测试函数"""
    formatter = PromptFormatter()
    
    # 测试新闻内容
    test_news = [
        "政府宣布将研究新的税收政策，具体方案尚未确定。",
        "央行表示正在评估货币政策工具的有效性。",
        "财政部召开会议讨论财政支出计划。"
    ]
    
    print("=== 测试提示词格式化 ===")
    
    # 测试单个EPU类型
    good_epu_prompt = formatter.get_epu_prompt('good_epu', test_news)
    print(f"向好EPU提示词长度: {len(good_epu_prompt)}")
    print(f"提示词预览: {good_epu_prompt[:200]}...")
    
    # 测试所有EPU类型
    all_prompts = formatter.get_all_epu_prompts(test_news)
    print(f"\n生成的提示词类型数量: {len(all_prompts)}")
    
    # 统计信息
    stats = formatter.get_prompt_statistics(all_prompts)
    print(f"\n提示词统计:")
    for epu_type, stat in stats.items():
        print(f"{epu_type}: 长度={stat['length']}, 有效={stat['valid']}")

if __name__ == "__main__":
    main()
