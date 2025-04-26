import pyaudio
from funasr import AutoModel
from ASR_methods import VadTest
from ASR_methods import GetAudio
from ASR_methods import Audio2Text
import os 
from LLM_modules import LLM

import sys

# 添加 src 目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 现在可以导入 f5_tts 下的 api 模块
from f5_tts import api  # 替换 some_function 为 api.py 中的实际函数名
tts=api.F5TTS()
print("TTS模型加载完成")







vad_model = VadTest.VAD_model()
asr_model = Audio2Text.ASR_Model_init()


# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512# 初始化 VAD 模型

flag=True

llm=LLM.llm_model()


while flag:
    try:
        # 创建 PyAudio 流
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        audio_file = GetAudio.get_audio_file(vad_model, stream)
        stream.stop_stream()
        stream.close()
        p.terminate()
        if audio_file:
            text=Audio2Text.get_text(asr_model, audio_file)
            print(f'识别结果:{text}')
            os.remove(audio_file)

            if "结束" in text:
                flag= False
                print("检测到结束语音，停止录音。")        
            print("回答开始：")
            for i in llm.chat(text):
                print(i,end="")
        
    except Exception as e:
        print(f"无法打开音频流: {e}")
        flag=False


