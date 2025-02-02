import os
import whisper
from pydub import AudioSegment
import pandas as pd
import re
import pyautogui
import time
import difflib  # Add this import

# ========== 配置参数 ==========
CUT_VIDEO_PATH = "/Users/maomao/Movies"  # 剪映项目路径
EXPORT_SRT_PATH = "/Users/maomao/Movies/字幕.srt"
ALIGNED_SRT_PATH = "/Users/maomao/Movies/aligned_subs.srt"
TTS_AUDIO_PATH = "/Users/maomao/Movies/音频.mp3"  # 假设AI语音已生成

# ========== 功能函数 ==========
def export_from_capcut():
    """自动化操作剪映导出字幕和音频"""
    # 1. 打开剪映并定位到项目
    pyautogui.hotkey('command', 'space')  # Mac打开Spotlight
    pyautogui.typewrite('剪映专业版')
    pyautogui.press('enter')
    time.sleep(5)
    
    # 2. 模拟导出操作（需根据实际UI调整坐标）
    pyautogui.click(x=100, y=200)  # 点击导出按钮
    pyautogui.click(x=150, y=300)  # 勾选导出字幕
    pyautogui.click(x=200, y=400)  # 确认导出路径
    time.sleep(2)

def get_speech_duration(text, all_words):
    """根据Whisper识别结果获取实际语音持续时间"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # 在所有词中寻找最匹配的片段
    best_duration = None
    best_ratio = 0
    
    for i in range(len(all_words)):
        for j in range(i + 1, min(i + 20, len(all_words) + 1)):
            candidate_words = all_words[i:j]
            candidate_text = ' '.join(word['word'] for word in candidate_words)
            candidate_text = re.sub(r'[^\w\s]', '', candidate_text.lower())
            
            ratio = difflib.SequenceMatcher(None, text, candidate_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                # 计算实际语音持续时间
                best_duration = candidate_words[-1]['end'] - candidate_words[0]['start']
    
    # 如果找到匹配，返回实际持续时间，否则返回保守估计
    return best_duration if best_duration and best_ratio > 0.2 else len(text) * 0.3

def align_subtitles(audio_path, srt_path, output_srt_path):
    """对齐字幕与AI语音时间轴"""
    # 1. 用Whisper识别语音时间戳
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # 2. 解析原字幕
    original_subs = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        subs = f.read().split('\n\n')
        for sub in subs:
            lines = sub.split('\n')
            if len(lines) < 3:
                continue
            text = lines[2].strip()
            time_parts = lines[1].split(' --> ')
            original_subs.append({
                'text': text,
                'start_time': _parse_time(time_parts[0]),
                'end_time': _parse_time(time_parts[1]),
                'duration': _parse_time(time_parts[1]) - _parse_time(time_parts[0])
            })
    
    # 3. 准备音频转录段落
    all_words = []
    total_duration = 0
    for segment in result["segments"]:
        if "words" in segment:
            all_words.extend(segment["words"])
            if segment["words"]:
                total_duration = max(total_duration, segment["words"][-1]["end"])
    
    # 4. 对齐处理
    aligned = []
    word_idx = 0
    MIN_GAP = 0.05
    total_subs = len(original_subs)
    
    for i, sub in enumerate(original_subs):
        # 对当前字幕文本进行归一化
        sub_text = re.sub(r'[^\w\s]', '', sub['text'].lower())
        sub_words = sub_text.split()
        
        # 计算搜索窗口大小
        remaining_subs = total_subs - i
        remaining_words = len(all_words) - word_idx
        avg_words_per_sub = remaining_words / max(1, remaining_subs)
        search_window = min(int(avg_words_per_sub * 2), remaining_words)
        
        # 在转录文本中查找最佳匹配
        best_match = None
        best_ratio = 0
        
        # 扩大搜索范围，确保不会漏掉匹配
        search_start = max(0, word_idx - len(sub_words))
        search_end = min(len(all_words), word_idx + search_window)
        
        for j in range(search_start, search_end):
            # 提取候选文本片段
            candidate_words = all_words[j:j + len(sub_words)]
            if not candidate_words:
                continue
            candidate_text = ' '.join(word['word'] for word in candidate_words)
            candidate_text = re.sub(r'[^\w\s]', '', candidate_text.lower())
            
            # 计算相似度
            ratio = difflib.SequenceMatcher(None, sub_text, candidate_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (j, j + len(candidate_words))
        
        # 获取实际语音持续时间
        speech_duration = get_speech_duration(sub['text'], all_words)
        # 添加缓冲时间（前后各0.2秒）
        min_duration = speech_duration + 0.4
        
        if best_match and best_ratio > 0.2:
            start_idx, end_idx = best_match
            new_start = all_words[start_idx]["start"]
            # 确保字幕持续时间不小于语音时长加缓冲
            new_end = new_start + max(min_duration, sub['duration'])
            word_idx = end_idx
        else:
            progress = i / total_subs
            new_start = progress * total_duration
            new_end = new_start + max(min_duration, sub['duration'])
        
        # 确保与前一个字幕有足够间隔
        if aligned and new_start < aligned[-1]["end"] + MIN_GAP:
            new_start = aligned[-1]["end"] + MIN_GAP
            new_end = new_start + max(min_duration, sub['duration'])
        
        aligned.append({
            "start": new_start,
            "end": new_end,
            "text": sub['text']
        })
    
    # 5. 生成新SRT
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(sorted(aligned, key=lambda x: x["start"]), 1):
            # 确保按开始时间排序并添加序号
            start = _format_time(sub["start"])
            end = _format_time(sub["end"])
            f.write(f"{idx}\n{start} --> {end}\n{sub['text']}\n\n")

def _parse_time(time_str):
    """解析SRT时间格式为秒数"""
    time_str = time_str.replace(',', '.')
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def _format_time(seconds):
    """时间格式转换"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    return f"{hours:02}:{minutes:02}:{sec:06.3f}".replace('.', ',')

# ========== 主流程 ==========
if __name__ == "__main__":
    # 1. 自动化导出数据（需先打开剪映项目）
    # export_from_capcut()
    
    # 2. 对齐字幕与语音
    align_subtitles(TTS_AUDIO_PATH, EXPORT_SRT_PATH, ALIGNED_SRT_PATH)
    
    # 3. 自动化导入对齐后的字幕
    # pyautogui.click(x=300, y=200)  # 点击剪映导入按钮
    # pyautogui.typewrite(ALIGNED_SRT_PATH)
    # pyautogui.press('enter')
    # print("对齐完成！")