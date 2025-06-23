"""
进度追踪工具
提供统一的进度显示和日志记录功能
"""
import time
import logging
from datetime import datetime
from typing import Optional, List

class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, total_steps: int, description: str = "处理中"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step_name: str, detail: str = ""):
        """更新进度"""
        self.current_step += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算百分比
        percentage = (self.current_step / self.total_steps) * 100
        
        # 估算剩余时间
        if self.current_step > 1:
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            remaining_str = f" (剩余约 {self._format_time(estimated_remaining)})"
        else:
            remaining_str = ""
        
        # 生成进度条
        bar_length = 50
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # 格式化输出
        progress_msg = (
            f"\n{'='*80}\n"
            f"🚀 {self.description} - 步骤 {self.current_step}/{self.total_steps} ({percentage:.1f}%)\n"
            f"📊 [{bar}] {percentage:.1f}%\n"
            f"⏱️  已用时间: {self._format_time(elapsed_time)}{remaining_str}\n"
            f"🔧 当前步骤: {step_name}\n"
        )
        
        if detail:
            progress_msg += f"📝 详细信息: {detail}\n"
        
        progress_msg += f"{'='*80}"
        
        print(progress_msg)
        logging.info(f"进度 {percentage:.1f}% - {step_name}: {detail}")
        
        # 记录步骤时间
        self.step_times.append({
            'step': self.current_step,
            'name': step_name,
            'time': current_time,
            'elapsed': elapsed_time
        })
    
    def complete(self, success: bool = True, final_message: str = ""):
        """完成进度追踪"""
        total_time = time.time() - self.start_time
        
        if success:
            status_icon = "✅"
            status_text = "完成"
        else:
            status_icon = "❌"
            status_text = "失败"
        
        completion_msg = (
            f"\n{'='*80}\n"
            f"{status_icon} {self.description} - {status_text}!\n"
            f"⏱️  总耗时: {self._format_time(total_time)}\n"
            f"📊 完成步骤: {self.current_step}/{self.total_steps}\n"
        )
        
        if final_message:
            completion_msg += f"📝 {final_message}\n"
        
        completion_msg += f"{'='*80}"
        
        print(completion_msg)
        logging.info(f"{self.description}完成 - 总耗时: {self._format_time(total_time)}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分钟"
    
    @staticmethod
    def create_subtask_tracker(total_subtasks: int, parent_description: str, subtask_name: str):
        """创建子任务追踪器"""
        description = f"{parent_description} > {subtask_name}"
        return ProgressTracker(total_subtasks, description)
    
    def log_milestone(self, milestone: str, details: Optional[List[str]] = None):
        """记录里程碑"""
        milestone_msg = (
            f"\n🎯 里程碑: {milestone}\n"
            f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        
        if details:
            milestone_msg += "📋 详细信息:\n"
            for detail in details:
                milestone_msg += f"   • {detail}\n"
        
        print(milestone_msg)
        logging.info(f"里程碑: {milestone}")
        
        if details:
            for detail in details:
                logging.info(f"  - {detail}")

# 全局进度追踪器实例
_global_tracker: Optional[ProgressTracker] = None

def init_global_tracker(total_steps: int, description: str):
    """初始化全局进度追踪器"""
    global _global_tracker
    _global_tracker = ProgressTracker(total_steps, description)
    return _global_tracker

def update_global_progress(step_name: str, detail: str = ""):
    """更新全局进度"""
    if _global_tracker:
        _global_tracker.update(step_name, detail)

def complete_global_progress(success: bool = True, final_message: str = ""):
    """完成全局进度追踪"""
    if _global_tracker:
        _global_tracker.complete(success, final_message)

def log_global_milestone(milestone: str, details: Optional[List[str]] = None):
    """记录全局里程碑"""
    if _global_tracker:
        _global_tracker.log_milestone(milestone, details) 