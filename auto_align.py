import os
import whisper
from pydub import AudioSegment
import pandas as pd
import re
import pyautogui
import time
import difflib  # Add this import
import datetime

# ========== 配置参数 ==========
CUT_VIDEO_PATH = "/Users/maomao/Movies"  # 剪映项目路径
EXPORT_SRT_PATH = "/Users/maomao/Movies/字幕.srt"
ALIGNED_SRT_PATH = "/Users/maomao/Movies/aligned_subs.srt"
TTS_AUDIO_PATH = "/Users/maomao/Movies/音频.mp3"  # 假设AI语音已生成

# ========== 功能函数 ==========
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

def process_audio(audio_path, processed_audio_path):
    """预处理音频，确保音频片段连续且不重叠"""
    try:
        # 读取音频文件
        audio = AudioSegment.from_file(audio_path)
        
        # 如果是多轨音频，将所有轨道混合成单轨
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 标准化音量
        audio = audio.normalize()
        
        # 确保采样率是16kHz (Whisper推荐的采样率)
        audio = audio.set_frame_rate(16000)
        
        # 导出处理后的音频
        audio.export(processed_audio_path, format="wav")
        return processed_audio_path
    except Exception as e:
        print(f"音频处理失败: {str(e)}")
        return audio_path

def process_subtitles(srt_path):
    """预处理字幕，处理重叠问题并确保时间顺序正确"""
    try:
        # 读取并解析字幕
        subs = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            current_sub = []
            for line in f:
                line = line.strip()
                if line:
                    current_sub.append(line)
                elif current_sub:
                    if len(current_sub) >= 3:  # 确保至少有序号、时间轴和文本
                        time_parts = current_sub[1].split(' --> ')
                        subs.append({
                            'index': int(current_sub[0]),
                            'start': _parse_time(time_parts[0]),
                            'end': _parse_time(time_parts[1]),
                            'text': '\n'.join(current_sub[2:])
                        })
                    current_sub = []
        
        # 按开始时间排序
        subs.sort(key=lambda x: x['start'])
        
        # 处理重叠问题
        MIN_GAP = 0.05  # 最小间隔时间（秒）
        processed_subs = []
        
        for i, sub in enumerate(subs):
            if i > 0:
                prev_sub = processed_subs[-1]
                # 如果当前字幕与前一个重叠
                if sub['start'] < prev_sub['end']:
                    # 不再合并字幕，只调整时间使其不重叠
                    sub['start'] = prev_sub['end'] + MIN_GAP
            
            processed_subs.append(sub)
        
        return processed_subs
    
    except Exception as e:
        print(f"字幕预处理失败: {str(e)}")
        return None

def align_subtitles(audio_path, srt_path, output_srt_path):
    """对齐字幕与AI语音时间轴"""
    # 1. 预处理音频
    processed_audio = os.path.join(os.path.dirname(audio_path), "processed_audio.wav")
    audio_path = process_audio(audio_path, processed_audio)
    
    # 2. 预处理字幕
    original_subs = process_subtitles(srt_path)
    if not original_subs:
        print("字幕预处理失败，退出程序")
        return
    
    # 3. 用Whisper识别语音时间戳
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # 4. 准备音频转录段落
    all_words = []
    total_duration = 0
    for segment in result["segments"]:
        if "words" in segment:
            all_words.extend(segment["words"])
            if segment["words"]:
                total_duration = max(total_duration, segment["words"][-1]["end"])
    
    # 5. 对齐处理
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
        # 调整字幕持续时间计算
        # 根据文本长度估算基础持续时间
        base_duration = len(sub['text']) * 0.25  # 每个字符大约0.25秒
        # 添加额外缓冲时间
        min_duration = max(base_duration, speech_duration) + 0.6  # 增加前后缓冲时间
        
        if best_match and best_ratio > 0.2:
            start_idx, end_idx = best_match
            new_start = all_words[start_idx]["start"]
            # 确保字幕持续时间足够长
            new_end = new_start + min_duration
            word_idx = end_idx
        else:
            progress = i / total_subs
            new_start = progress * total_duration
            new_end = new_start + min_duration
        
        # 确保与前一个字幕有足够间隔
        if aligned and new_start < aligned[-1]["end"] + MIN_GAP:
            new_start = aligned[-1]["end"] + MIN_GAP
            new_end = new_start + max(min_duration, sub['end'] - sub['start'])
        
        aligned.append({
            "start": new_start,
            "end": new_end,
            "text": sub['text']
        })
    
    # 6. 生成新SRT
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(sorted(aligned, key=lambda x: x["start"]), 1):
            # 确保按开始时间排序并添加序号
            start = _format_time(sub["start"])
            end = _format_time(sub["end"])
            f.write(f"{idx}\n{start} --> {end}\n{sub['text']}\n\n")

    # 清理临时文件
    if os.path.exists(processed_audio) and processed_audio != audio_path:
        os.remove(processed_audio)

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
    # 1. 对齐字幕与语音
    align_subtitles(TTS_AUDIO_PATH, EXPORT_SRT_PATH, ALIGNED_SRT_PATH)