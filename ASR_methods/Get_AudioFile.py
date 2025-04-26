import pyaudio
import ASR_methods.VAD_test as VAD_test
import time
import numpy as np
import soundfile as sf
from datetime import datetime

class VAD_audio:
    def __init__(self):
        # 实例化 VAD_model 类
        self.vad = VAD_test.VAD_model()

        # 音频流参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.SAMPLE_RATE = 16000
        self.CHUNK = 512  # 每帧 512 样本



    def get_audio(self,chunk):
        # 状态变量
        is_recording = False  # 是否正在录音
        silence_start_time = None  # 沉默开始时间
        SILENCE_THRESHOLD = 1.5  # 沉默阈值（1.5 秒）
        audio_data = []  # 保存所有音频块

        try:
            print("开始检测语音（按 Ctrl+C 停止）...")
            while True:
                # 读取音频块
                audio_chunk = chunk

                # 使用 VAD 检测语音
                is_speech = self.vad.VAD_test(audio_chunk, self.SAMPLE_RATE)

                if is_speech:
                    # 检测到语音
                    if not is_recording:
                        # 第一次检测到语音，开始录音
                        print("检测到语音，开始记录音频文件...")
                        is_recording = True
                    # 重置沉默开始时间
                    silence_start_time = None
                    audio_data.append(audio_chunk)
                else:
                    # 检测到沉默
                    if is_recording:
                        # 正在录音中，检测到沉默
                        if silence_start_time is None:
                            # 第一次检测到沉默，记录开始时间
                            silence_start_time = time.time()
                            audio_data.append(audio_chunk)
                        else:
                            audio_data.append(audio_chunk)
                            # 已经开始计时，计算沉默持续时间
                            silence_duration = time.time() - silence_start_time
                            if silence_duration >= SILENCE_THRESHOLD:
                                # 沉默超过阈值，结束录音
                                print(f"检测到 {SILENCE_THRESHOLD} 秒沉默，结束录音。")
                                break
                    # 如果未开始录音（is_recording=False），继续等待语音

        except KeyboardInterrupt:
            print("停止检测（用户中断）。")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 保存音频文件
            if audio_data and is_recording:
                # 计算每个块的时长（秒）
                chunk_duration = self.CHUNK / self.SAMPLE_RATE  # 每个块的时长（秒）
                num_silence_chunks = int(SILENCE_THRESHOLD / chunk_duration)
                # 截断 audio_data，移除最后的沉默块
                if len(audio_data) > num_silence_chunks:
                    audio_data = audio_data[:-num_silence_chunks]
                else:
                    audio_data = []  # 如果数据不足 SILENCE_THRESHOLD 秒，清空

                # 确保 audio_data 不为空
                if audio_data:
                    # 将所有音频块拼接为一个数组
                    audio_array = np.concatenate([np.frombuffer(chunk, np.int16) for chunk in audio_data])
                    # 使用时间戳生成唯一文件名
                    output_file = f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    sf.write(output_file, audio_array, self.SAMPLE_RATE)
                    print(f"音频已保存到: {output_file}（已移除最后 {SILENCE_THRESHOLD} 秒沉默）")
                    return output_file
                else:
                    print("未录制到有效语音（录音时长不足）。")
                    return None
            else:
                print("未录制到音频。")
                return None

# # 使用示例
# if __name__ == "__main__":
#     vad_audio = VAD_audio()
#     try:
#         audio_file = vad_audio.get_audio()
#         if audio_file:
#             print(f"音频文件已提交: {audio_file}")
#     finally:
#         del vad_audio  # 确保资源清理