#调用silero-vad模型进行VAD检测


import numpy as np
import torch

# 设置 PyTorch 线程数
torch.set_num_threads(1)

class VAD_model():
    def __init__(self):
        model_path = ".//models//silero_vad//silero_vad.jit"  # 替换为你的实际路径
        try:
            model = torch.jit.load(model_path)
            print(f"成功加载本地模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 如果本地加载失败，尝试在线加载
            model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )        
        self.model=model

    # 将 int16 格式音频转换为 float32 格式
    def int2float(self,sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768  # 归一化到 [-1, 1]
        sound = sound.squeeze()
        return sound

    # 简化录制和检测函数
    def VAD_test(self,audio_chunk, SAMPLE_RATE=16000):    
        try:
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)
            # 使用 Silero VAD 模型计算语音概率
            confidence = self.model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
            # 输出结果
            if confidence > 0.5:  # 语音概率阈值
                return True
            else: return False


        except KeyboardInterrupt:
            print("停止检测。")


