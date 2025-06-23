"""
DeepSeek API并行调用模块
使用多个API密钥并行调用DeepSeek API生成EPU分析结果
"""
import asyncio
import aiohttp
import json
import logging
import time
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from config.deepseek_api_keys import get_all_api_keys, get_api_key
from src.epu_generator.format_prompt import PromptFormatter

class DeepSeekParallelRunner:
    """DeepSeek API并行调用器"""
    
    def __init__(self):
        self.api_keys = get_all_api_keys()
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.max_concurrent = min(len(self.api_keys), Config.MAX_CONCURRENT_REQUESTS)
        self.timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
        self.retry_times = Config.RETRY_TIMES
        self.prompt_formatter = PromptFormatter()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_count': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def call_deepseek_api(self, session, prompt, api_key_index, request_id):
        """调用单个DeepSeek API"""
        api_key = self.api_keys[api_key_index % len(self.api_keys)]
        
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
        
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(self.retry_times):
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # 提取回复内容
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message']['content'].strip()
                            
                            # 尝试提取数字分数
                            score = self.extract_score(content)
                            
                            return {
                                'success': True,
                                'score': score,
                                'raw_response': content,
                                'api_key_index': api_key_index,
                                'request_id': request_id,
                                'attempt': attempt + 1
                            }
                        else:
                            logging.warning(f"API响应格式异常: {result}")
                            
                    else:
                        error_text = await response.text()
                        logging.warning(f"API请求失败 (尝试 {attempt + 1}): HTTP {response.status}, {error_text}")
                        
                        # 如果是429错误（请求过多），等待更长时间
                        if response.status == 429:
                            wait_time = (attempt + 1) * 2
                            await asyncio.sleep(wait_time)
                            
            except Exception as e:
                logging.warning(f"API请求异常 (尝试 {attempt + 1}): {str(e)}")
                if attempt < self.retry_times - 1:
                    wait_time = (attempt + 1) * 1
                    await asyncio.sleep(wait_time)
                    self.stats['retry_count'] += 1
        
        # 所有重试都失败
        return {
            'success': False,
            'score': 0,
            'raw_response': '',
            'api_key_index': api_key_index,
            'request_id': request_id,
            'error': 'All retries failed'
        }
    
    def extract_score(self, response_text):
        """从响应文本中提取分数"""
        import re
        
        # 查找数字模式
        patterns = [
            r'输出[:：]\s*(\d+)',
            r'分数[:：]\s*(\d+)',
            r'评分[:：]\s*(\d+)',
            r'(\d+)分',
            r'(\d+)$',  # 行末的数字
            r'^(\d+)',  # 行首的数字
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                try:
                    score = int(matches[0])
                    if 0 <= score <= 100:
                        return score
                except ValueError:
                    continue
        
        # 如果没有找到有效分数，尝试更宽泛的数字搜索
        numbers = re.findall(r'\d+', response_text)
        for num_str in numbers:
            try:
                num = int(num_str)
                if 0 <= num <= 100:
                    return num
            except ValueError:
                continue
        
        # 如果仍然没有找到，返回默认值
        logging.warning(f"无法从响应中提取分数: {response_text}")
        return 0
    
    async def process_requests_batch(self, requests_data):
        """批量处理API请求"""
        self.stats['total_requests'] = len(requests_data)
        self.stats['start_time'] = datetime.now()
        
        # 创建session
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        ) as session:
            
            # 创建任务列表
            tasks = []
            for i, request_data in enumerate(requests_data):
                api_key_index = i % len(self.api_keys)
                task = self.call_deepseek_api(
                    session, 
                    request_data['prompt'], 
                    api_key_index, 
                    i
                )
                tasks.append(task)
                
                # 添加请求间延迟
                if i > 0 and i % self.max_concurrent == 0:
                    await asyncio.sleep(1)
            
            # 并行执行所有任务
            logging.info(f"开始并行执行 {len(tasks)} 个API请求")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"请求 {i} 执行异常: {str(result)}")
                    self.stats['failed_requests'] += 1
                    
                    # 创建失败结果
                    processed_result = {
                        'success': False,
                        'score': 0,
                        'raw_response': '',
                        'error': str(result),
                        'request_data': requests_data[i]
                    }
                else:
                    if result['success']:
                        self.stats['successful_requests'] += 1
                    else:
                        self.stats['failed_requests'] += 1
                    
                    # 添加原始请求数据
                    processed_result = result.copy()
                    processed_result['request_data'] = requests_data[i]
                
                processed_results.append(processed_result)
        
        self.stats['end_time'] = datetime.now()
        return processed_results
    
    def save_results(self, results, output_dir):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 按日期和EPU类型组织结果
        results_by_date = {}
        
        for result in results:
            request_data = result['request_data']
            date = request_data['date']
            epu_type = request_data['epu_type']
            
            if date not in results_by_date:
                results_by_date[date] = {}
            
            results_by_date[date][epu_type] = {
                'epu_type': epu_type,
                'epu_type_name': request_data['epu_type_name'],
                'score': result['score'],
                'success': result['success'],
                'raw_response': result['raw_response'],
                'content_count': request_data['content_count'],
                'process_time': datetime.now().isoformat()
            }
        
        # 保存每日结果
        for date, date_results in results_by_date.items():
            output_file = os.path.join(output_dir, f"epu_results_{date}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(date_results, f, ensure_ascii=False, indent=2)
            
            logging.info(f"保存EPU结果: {output_file}")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "processing_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2, default=str)
        
        logging.info(f"保存处理统计: {stats_file}")
        
        return results_by_date
    
    def get_statistics(self):
        """获取处理统计信息"""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            self.stats['duration_seconds'] = duration
            self.stats['requests_per_second'] = self.stats['total_requests'] / duration if duration > 0 else 0
        
        return self.stats
    
    def process_news_data(self, news_data_list, output_dir):
        """处理新闻数据并生成EPU结果"""
        # 准备API请求数据
        logging.info("准备API请求数据...")
        requests_data = self.prompt_formatter.prepare_api_requests(news_data_list)
        
        if not requests_data:
            logging.warning("没有有效的请求数据")
            return None
        
        logging.info(f"准备了 {len(requests_data)} 个API请求")
        
        # 并行处理请求
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(self.process_requests_batch(requests_data))
        finally:
            loop.close()
        
        # 保存结果
        results_by_date = self.save_results(results, output_dir)
        
        # 输出统计信息
        stats = self.get_statistics()
        logging.info(f"处理完成 - 总请求: {stats['total_requests']}, 成功: {stats['successful_requests']}, 失败: {stats['failed_requests']}")
        logging.info(f"处理时间: {stats.get('duration_seconds', 0):.2f}秒, 速度: {stats.get('requests_per_second', 0):.2f}请求/秒")
        
        return results_by_date

def load_merged_news_data(start_date, end_date):
    """加载合并后的新闻数据"""
    from datetime import datetime, timedelta
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    news_data_list = []
    current_dt = start_dt
    
    merged_dir = os.path.join(Config.RAW_NEWS_DIR, 'merged')
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')
        merged_file = os.path.join(merged_dir, f"{date_str}_merged.json")
        
        if os.path.exists(merged_file):
            try:
                with open(merged_file, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                    news_data_list.append(news_data)
            except Exception as e:
                logging.error(f"读取合并新闻数据失败 {merged_file}: {str(e)}")
        
        current_dt += timedelta(days=1)
    
    return news_data_list

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 确保目录存在
    Config.ensure_directories()
    
    # 创建DeepSeek并行运行器
    runner = DeepSeekParallelRunner()
    
    # 加载新闻数据
    start_date = Config.START_DATE
    end_date = Config.END_DATE
    
    logging.info(f"开始加载新闻数据: {start_date} 到 {end_date}")
    news_data_list = load_merged_news_data(start_date, end_date)
    
    if not news_data_list:
        logging.error("没有找到有效的新闻数据")
        return
    
    logging.info(f"加载了 {len(news_data_list)} 天的新闻数据")
    
    # 处理EPU分析
    output_dir = Config.RAW_EPU_RESULTS_DIR
    results = runner.process_news_data(news_data_list, output_dir)
    
    if results:
        logging.info(f"EPU处理完成，生成了 {len(results)} 天的结果")
    else:
        logging.error("EPU处理失败")

if __name__ == "__main__":
    main()
