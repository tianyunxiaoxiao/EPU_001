"""
人民日报新闻收集器
从人民日报网站获取每日新闻内容
"""
import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
import logging
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class PeopleDailyCollector:
    """人民日报新闻收集器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def collect_daily_news(self, date_str):
        """收集指定日期的新闻
        
        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            
        Returns:
            dict: 包含新闻内容的字典
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            year = date_obj.year
            month = f"{date_obj.month:02d}"
            day = f"{date_obj.day:02d}"
            
            # 构建人民日报URL
            url = f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/nbs.D110000renmrb_01.htm"
            
            logging.info(f"正在收集人民日报新闻: {url}")
            
            response = self.session.get(url, timeout=30)
            
            # 如果主页不存在，尝试其他版面
            if response.status_code == 404:
                # 尝试不同的版面链接
                alternative_urls = [
                    f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/nw.D110000renmrb_01.htm",
                    f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/",
                ]
                
                for alt_url in alternative_urls:
                    try:
                        response = self.session.get(alt_url, timeout=30)
                        if response.status_code == 200:
                            url = alt_url
                            break
                    except:
                        continue
                        
                if response.status_code != 200:
                    logging.warning(f"无法访问人民日报页面: {date_str}")
                    return None
            
            response.raise_for_status()
            response.encoding = 'gb2312'  # 人民日报网站编码
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取新闻标题和链接
            news_content = []
            
            # 查找新闻链接
            news_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in news_links:
                href = link.get('href')
                if href and 'nw.D110000renmrb' in href:
                    full_url = f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/{href}"
                    article_urls.append(full_url)
                    
                    # 提取标题
                    title = link.get_text().strip()
                    if title and len(title) > 5:
                        news_content.append(f"标题: {title}")
            
            # 限制文章数量避免过多请求
            article_urls = article_urls[:10]
            
            # 获取文章内容
            for article_url in article_urls:
                try:
                    article_content = self.get_article_content(article_url)
                    if article_content:
                        news_content.extend(article_content)
                    time.sleep(0.5)  # 延迟
                except:
                    continue
            
            if not news_content:
                logging.warning(f"未找到有效新闻内容: {date_str}")
                return None
                
            news_data = {
                'date': date_str,
                'source': 'people_daily',
                'url': url,
                'content': news_content,
                'collect_time': datetime.now().isoformat(),
                'content_count': len(news_content)
            }
            
            logging.info(f"成功收集人民日报新闻: {date_str}, 内容条数: {len(news_content)}")
            return news_data
            
        except Exception as e:
            logging.error(f"收集人民日报新闻失败 {date_str}: {str(e)}")
            return None
    
    def get_article_content(self, article_url):
        """获取单篇文章内容"""
        try:
            response = self.session.get(article_url, timeout=20)
            response.encoding = 'gb2312'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 查找文章内容区域
            content_areas = [
                soup.find('div', {'id': 'ozoom'}),
                soup.find('div', class_='show_text'),
                soup.find('div', class_='content'),
                soup.find('td', class_='show_text')
            ]
            
            article_content = []
            
            for content_area in content_areas:
                if content_area:
                    paragraphs = content_area.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 20:
                            article_content.append(text)
                    break
            
            return article_content
            
        except Exception as e:
            logging.debug(f"获取文章内容失败: {article_url}, {str(e)}")
            return []
    
    def collect_date_range(self, start_date, end_date):
        """收集指定日期范围的新闻"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_news = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # 检查是否已存在文件 - 按分层目录检查
            date_obj = current_dt
            year_dir = os.path.join(Config.RAW_NEWS_DIR, str(date_obj.year))
            month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
            day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
            output_file = os.path.join(day_dir, f"people_daily_{date_str}.json")
            
            if os.path.exists(output_file):
                logging.info(f"跳过已存在的文件: {date_str}")
                current_dt += timedelta(days=1)
                continue
            
            news_data = self.collect_daily_news(date_str)
            
            if news_data:
                self.save_daily_news(news_data)
                all_news.append(news_data)
            
            # 添加延迟避免过于频繁请求
            time.sleep(2)
            
            current_dt += timedelta(days=1)
        
        return all_news
    
    def save_daily_news(self, news_data):
        """保存单日新闻数据 - 按天-月-年分层保存"""
        if not news_data:
            return
            
        date_str = news_data['date']
        
        # 创建分层目录结构 year/month/day/
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year_dir = os.path.join(Config.RAW_NEWS_DIR, str(date_obj.year))
        month_dir = os.path.join(year_dir, f"{date_obj.month:02d}")
        day_dir = os.path.join(month_dir, f"{date_obj.day:02d}")
        
        os.makedirs(day_dir, exist_ok=True)
        
        # 保存JSON文件
        output_file = os.path.join(day_dir, f"people_daily_{date_str}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"保存人民日报新闻: {output_file}")

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 确保目录存在
    Config.ensure_directories()
    
    collector = PeopleDailyCollector()
    
    # 收集指定时间范围的新闻
    start_date = Config.START_DATE
    end_date = Config.END_DATE
    
    logging.info(f"开始收集人民日报新闻: {start_date} 到 {end_date}")
    
    news_list = collector.collect_date_range(start_date, end_date)
    
    logging.info(f"收集完成，总计 {len(news_list)} 天的新闻")

if __name__ == "__main__":
    main()
