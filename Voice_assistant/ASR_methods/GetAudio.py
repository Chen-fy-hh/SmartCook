import torch
import numpy as np
import pyaudio
import time
from datetime import datetime
import soundfile as sf
from typing import Optional, Any

class AudioRecorder:
    def __init__(self,vad_model: Any):

        self.vad_model = vad_model
        self.SAMPLE_RATE = 16000
        self.CHUNK = 512
        self.SILENCE_THRESHOLD = 1.5  # 沉默阈值（秒）

    def record_audio(self, stream: pyaudio.Stream) -> Optional[str]:
        """
        从 PyAudio 流中录制音频，使用 VAD 检测语音，返回音频文件路径
        :param stream: 已打开的 PyAudio 流
        :return: 录制的音频文件路径，或 None（如果未录制到有效音频）
        """
        # 状态变量
        is_recording = False
        silence_start_time = None
        audio_data = []

        print("开始检测语音（按 Ctrl+C 停止）...")

        try:
            while True:
                # 读取音频块
                audio_chunk = stream.read(self.CHUNK)
                # 使用 VAD 检测语音
                is_speech = self.vad_model.VAD_test(audio_chunk)

                if is_speech:
                    # 检测到语音
                    if not is_recording:
                        print("检测到语音，开始记录音频文件...")
                        is_recording = True
                    silence_start_time = None
                    audio_data.append(audio_chunk)
                else:
                    # 检测到沉默
                    if is_recording:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                            audio_data.append(audio_chunk)
                        else:
                            audio_data.append(audio_chunk)
                            silence_duration = time.time() - silence_start_time
                            if silence_duration >= self.SILENCE_THRESHOLD:
                                print(f"检测到 {self.SILENCE_THRESHOLD} 秒沉默，结束录音。")
                                break

        except KeyboardInterrupt:
            print("停止检测（用户中断）。")
        except Exception as e:
            print(f"发生错误: {e}")

        # 保存音频文件
        if audio_data and is_recording:
            # 计算每个块的时长（秒）
            chunk_duration = self.CHUNK / self.SAMPLE_RATE
            num_silence_chunks = int(self.SILENCE_THRESHOLD / chunk_duration)
            # 截断最后的沉默块
            if len(audio_data) > num_silence_chunks:
                audio_data = audio_data[:-num_silence_chunks]
            else:
                audio_data = []

            if audio_data:
                # 拼接音频数据
                audio_array = np.concatenate([np.frombuffer(chunk, np.int16) for chunk in audio_data])
                # 生成文件名
                output_file = f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                sf.write(output_file, audio_array, self.SAMPLE_RATE)
                print(f"音频已保存到: {output_file}（已移除最后 {self.SILENCE_THRESHOLD} 秒沉默）")
                return output_file
            else:
                print("未录制到有效语音（录音时长不足）。")
                return None
        else:
            print("未录制到音频。")
            return None

def get_audio_file(vad_model: Any, stream: pyaudio.Stream) -> Optional[str]:
    recorder = AudioRecorder(vad_model)
    return recorder.record_audio(stream)

