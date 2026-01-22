# python test.py

# 测试30s识别
import dolphin

waveform = dolphin.load_audio("D:/_Engineering/Leet/LeetSYS/AI/DolphinASR/data/datasets/G00005/foo.wav")
model = dolphin.load_model("base", "D:\_Engineering\Leet\LeetSYS\AI\DolphinASR\data\models\dolphin", "cpu")
result = model(waveform)

# Specify language
result = model(waveform, lang_sym="zh")

print(result.text)


# 测试>30s识别
'''
import dolphin
from dolphin.audio import convert_audio
from pathlib import Path
import os
import pydub
import numpy as np
import re
import difflib

# Load model on CPU, enable length normalization to reduce repetition
model = dolphin.load_model("base", "D:\_Engineering\Leet\LeetSYS\AI\DolphinASR\data\models\dolphin", "cpu", normalize_length=True)

audio_path = "D:/_Engineering/Leet/LeetSYS/AI/DolphinASR/data/datasets/G00005/foo.wav"
segments_out = Path("transcript_segments.txt")
full_out = Path("transcript_full.txt")

# 压缩重复字符，避免异常重复（如“定定定...”）
def compress_repetitions(text: str, max_repeat: int = 3) -> str:
    return re.sub(r"(.)\1{" + str(max_repeat) + ",}", lambda m: m.group(1) * max_repeat, text)
 
def fmt_mmss(ms: int) -> str:
    m = ms // 60000
    s = (ms % 60000) // 1000
    return f"{m:02d}:{s:02d}"
 
# 固定窗口 + 重叠：替代 VAD 的长语音分段，避免边界漏字
def transcribe_sliding_windows(model, wav_path, lang_sym, region_sym, window_s=30, hop_s=10, segments_out_path: Path = None, step_s=30, stream_non_overlap=True):
    tmp_audio = f"{wav_path}.tmp16k.wav"
    convert_audio(wav_path, tmp_audio)
    audio_seg = pydub.AudioSegment.from_wav(tmp_audio)
    duration_ms = len(audio_seg)
    window_ms = int(window_s * 1000)
    hop_ms = int(hop_s * 1000)
    parts = []  # (start_ms, end_ms, text)
    step_ms = int(step_s * 1000)
    start = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        chunk = audio_seg[start:end]
        raw = chunk.raw_data
        waveform = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        if waveform.size == 0 or np.max(np.abs(waveform)) < 1e-3:
            clean_text = ""
            parts.append((start, end, clean_text))
        else:
            result = model(
                speech=waveform.flatten(),
                lang_sym=lang_sym,
                region_sym=region_sym,
                predict_time=True,
                padding_speech=False,
            )
            clean_text = compress_repetitions(result.text_nospecial)
            parts.append((start, end, clean_text))
        # 仅输出非重叠片段（例如 0-30、30-60 …）
        if stream_non_overlap and (start % step_ms == 0):
            line = f"[{fmt_mmss(start)} - {fmt_mmss(end)}] {clean_text}"
            print(line, flush=True)
            if segments_out_path is not None:
                with segments_out_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        # 不要在触达末尾时提前中断，否则会错过最后一个非重叠锚点（如 90-108s）
        # 保持按 hop 递增直到 start >= duration_ms
        start += hop_ms
    Path(tmp_audio).unlink(missing_ok=True)
    return parts

# 运行滑动窗口长转写（30s窗口，10s跳步，重叠20s），区域保持 TW 或 CN 按需
# 先清空分段文件，随后在识别过程中逐步追加写入
with segments_out.open("w", encoding="utf-8"):
    pass
fixed_parts = transcribe_sliding_windows(model, audio_path, "zh", "CN", window_s=30, hop_s=10, segments_out_path=segments_out, step_s=30, stream_non_overlap=True)
segments = [
    type("Seg", (), {
        "start": s/1000.0,
        "end": e/1000.0,
        "text_nospecial": t,
    })() for (s, e, t) in fixed_parts
]

# 合并重叠窗口的文本，减少重复并补全句子
def merge_with_overlap(prev: str, curr: str) -> str:
    prev_clean = prev.strip()
    curr_clean = curr.strip()
    if not prev_clean:
        return curr_clean
    tail = prev_clean[-80:]
    head = curr_clean[:80]
    matcher = difflib.SequenceMatcher(None, tail, head)
    match = matcher.find_longest_match(0, len(tail), 0, len(head))
    if match.size >= 10:
        return prev_clean + curr_clean[match.b+match.size:]
    return prev_clean + curr_clean

# 写入本地文本文件（全文）

full_text = ""
for _, _, t in fixed_parts:
    clean_text = compress_repetitions(t)
    full_text = merge_with_overlap(full_text, clean_text)
with full_out.open("w", encoding="utf-8") as f:
    f.write(full_text)

print(f"已写入分段文本: {segments_out}")
print(f"已写入全文拼接: {full_out}")
'''