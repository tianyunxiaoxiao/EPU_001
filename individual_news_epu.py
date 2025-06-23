"""
逐个新闻的EPU生成器
为每条新闻生成详细的EPU分析结果
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.epu_generator.run_deepseek_parallel import DeepSeekParallelRunner
from src.epu_generator.format_prompt import EPUPromptFormatter

class IndividualNewsEPUGenerator:
    """逐个新闻的EPU生成器"""
    
    def __init__(self):
        self.deepseek_runner = DeepSeekParallelRunner()
        self.prompt_formatter = EPUPromptFormatter()
        
    async def _call_single_api(self, prompt):
        """
        调用单个DeepSeek API
        
        Args:
            prompt: 提示词
            
        Returns:
            dict: API响应结果
        """
        try:
            import aiohttp
            from config.deepseek_api_keys import get_api_key
            
            api_key = get_api_key()
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.1,
                'max_tokens': 100,
                'stream': False
            }
            
            url = f"{Config.DEEPSEEK_BASE_URL}/chat/completions"
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 提取回复内容
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content'].strip()
                            
                            # 提取分数
                            score = self.deepseek_runner.extract_score(content)
                            
                            return {
                                'success': True,
                                'score': score,
                                'raw_response': content
                            }
                        else:
                            return {
                                'success': False,
                                'score': 0,
                                'raw_response': ''
                            }
                    else:
                        error_text = await response.text()
                        logging.error(f"API请求失败: HTTP {response.status}, {error_text}")
                        return {
                            'success': False,
                            'score': 0,
                            'raw_response': f'HTTP {response.status}: {error_text}'
                        }
                        
        except Exception as e:
            logging.error(f"API调用异常: {str(e)}")
            return {
                'success': False,
                'score': 0,
                'raw_response': f'异常: {str(e)}'
            }
        
    def process_single_news_item(self, news_text, date_str, source="people_daily"):
        """
        处理单条新闻，生成EPU分析
        
        Args:
            news_text: 新闻文本
            date_str: 日期字符串 YYYY-MM-DD
            source: 新闻来源
            
        Returns:
            dict: 包含新闻文本和各种EPU分析结果
        """
        try:
            # 格式化提示词
            prompts = self.prompt_formatter.format_all_prompts(news_text)
            
            # 获取所有EPU类型的结果
            results = {}
            
            for epu_type, prompt in prompts.items():
                try:
                    # 使用DeepSeek API生成EPU得分
                    import asyncio
                    import aiohttp
                    
                    response = asyncio.run(self._call_single_api(prompt))
                    if response and response.get('success'):
                        results[f"{epu_type}_score"] = response['score']
                        results[f"{epu_type}_reasoning"] = response.get('raw_response', '')
                    else:
                        results[f"{epu_type}_score"] = None
                        results[f"{epu_type}_reasoning"] = 'API调用失败'
                        
                    # 添加延迟避免API限制
                    time.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"处理EPU类型 {epu_type} 失败: {str(e)}")
                    results[f"{epu_type}_score"] = None
                    results[f"{epu_type}_reasoning"] = f'处理错误: {str(e)}'
            
            # 构建完整结果
            result = {
                'date': date_str,
                'source': source,
                'news_text': news_text,
                'news_length': len(news_text),
                'process_time': datetime.now().isoformat(),
                **results
            }
            
            return result
            
        except Exception as e:
            logging.error(f"处理新闻失败: {str(e)}")
            return {
                'date': date_str,
                'source': source,
                'news_text': news_text,
                'news_length': len(news_text),
                'process_time': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def process_daily_news(self, date_str):
        """
        处理某一天的所有新闻
        
        Args:
            date_str: 日期字符串 YYYY-MM-DD
            
        Returns:
            list: 包含所有新闻EPU分析结果的列表
        """
        try:
            # 根据分层目录结构查找新闻文件
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year_dir = os.path.join(Config.RAW_NEWS_DIR, str(date_obj.year))
            month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
            day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
            news_file = os.path.join(day_dir, f"people_daily_{date_str}.json")
            
            if not os.path.exists(news_file):
                logging.warning(f"新闻文件不存在: {news_file}")
                return []
            
            # 加载新闻数据
            with open(news_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            
            if 'content' not in news_data or not news_data['content']:
                logging.warning(f"新闻文件无内容: {date_str}")
                return []
            
            # 处理每条新闻
            results = []
            content_list = news_data['content']
            
            logging.info(f"开始处理 {date_str} 的 {len(content_list)} 条新闻")
            
            # 使用线程池并行处理
            max_workers = min(5, len(content_list))  # 限制并发数
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_news = {
                    executor.submit(
                        self.process_single_news_item, 
                        news_text, 
                        date_str, 
                        news_data.get('source', 'people_daily')
                    ): news_text 
                    for news_text in content_list
                }
                
                # 收集结果
                for future in as_completed(future_to_news):
                    try:
                        result = future.result(timeout=60)  # 60秒超时
                        if result:
                            results.append(result)
                    except Exception as e:
                        news_text = future_to_news[future]
                        logging.error(f"处理新闻失败: {news_text[:50]}... 错误: {str(e)}")
            
            logging.info(f"完成处理 {date_str}，成功处理 {len(results)} 条新闻")
            return results
            
        except Exception as e:
            logging.error(f"处理日期 {date_str} 失败: {str(e)}")
            return []
    
    def save_daily_results(self, results, date_str):
        """
        保存单日的EPU分析结果 - 按天-月-年分层保存
        
        Args:
            results: EPU分析结果列表
            date_str: 日期字符串 YYYY-MM-DD
        """
        if not results:
            return
        
        try:
            # 创建分层目录结构 year/month/day/
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year_dir = os.path.join(Config.EPU_OUTPUT_DIR, "individual_news", str(date_obj.year))
            month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
            day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
            
            os.makedirs(day_dir, exist_ok=True)
            
            # 保存为CSV格式
            csv_file = os.path.join(day_dir, f"individual_news_epu_{date_str}.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            # 保存为JSON格式 (保留完整信息)
            json_file = os.path.join(day_dir, f"individual_news_epu_{date_str}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"逐个新闻EPU结果已保存: {day_dir}")
            logging.info(f"CSV文件: {csv_file}")
            logging.info(f"JSON文件: {json_file}")
            
        except Exception as e:
            logging.error(f"保存EPU结果失败 {date_str}: {str(e)}")
    
    def process_date_range(self, start_date, end_date):
        """
        处理指定日期范围的所有新闻
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            
        Returns:
            dict: 统计信息
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_processed = 0
        total_days = 0
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # 检查是否已存在结果文件
            date_obj = current_dt
            year_dir = os.path.join(Config.EPU_OUTPUT_DIR, "individual_news", str(date_obj.year))
            month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
            day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
            result_file = os.path.join(day_dir, f"individual_news_epu_{date_str}.csv")
            
            if os.path.exists(result_file):
                logging.info(f"跳过已存在的EPU结果: {date_str}")
                current_dt += timedelta(days=1)
                continue
            
            # 处理这一天的新闻
            results = self.process_daily_news(date_str)
            
            if results:
                self.save_daily_results(results, date_str)
                total_processed += len(results)
                total_days += 1
            
            current_dt += timedelta(days=1)
        
        stats = {
            'total_days_processed': total_days,
            'total_news_processed': total_processed,
            'average_news_per_day': total_processed / max(total_days, 1)
        }
        
        logging.info(f"逐个新闻EPU处理完成: {stats}")
        return stats

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 确保目录存在
    Config.ensure_directories()
    
    generator = IndividualNewsEPUGenerator()
    
    # 处理指定时间范围
    start_date = Config.START_DATE
    end_date = Config.END_DATE
    
    logging.info(f"开始生成逐个新闻EPU分析: {start_date} 到 {end_date}")
    
    stats = generator.process_date_range(start_date, end_date)
    
    logging.info(f"逐个新闻EPU生成完成: {stats}")

if __name__ == "__main__":
    main() 