# -*- encoding: utf-8 -*-
import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import sys
import os
import numpy as np
import wave
import tempfile
import time
from typing import List, Dict, Any
from ASR_methods import Audio2Text, VadTest, GetAudio
from LLM_modules import LLM
import sys
import base64
# 添加 src 目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 现在可以导入 f5_tts 下的 api 模块
from f5_tts import api  # 替换 some_function 为 api.py 中的实际函数名


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化模型
vad_model = VadTest.VAD_model()
asr_model = Audio2Text.ASR_Model_init()
llm = LLM.llm_model()
tts=api.F5TTS()
print("模型加载完成")
# 音频参数
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 512

# FastAPI 应用
app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        await websocket.send_json(data)

manager = ConnectionManager()

class WebSocketAudioProcessor:
    def __init__(self, websocket, manager, vad_model, temp_dir=None):
        self.websocket = websocket
        self.manager = manager
        self.vad_model = vad_model
        self.SAMPLE_RATE = 16000
        self.CHUNK = 512
        self.SILENCE_THRESHOLD = 1.5  # 沉默阈值（秒）
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        
        self.is_recording = False
        self.silence_start_time = None
        self.audio_data = []

    async def process_audio_chunk(self, audio_chunk):
        is_speech = self.vad_model.VAD_test(audio_chunk.tobytes())
        current_time = time.time()
        
        if is_speech:
            if not self.is_recording:
                logger.info("检测到语音，开始录音。")
                self.is_recording = True
            self.silence_start_time = None
            self.audio_data.append(audio_chunk)
            return None
        else:
            if self.is_recording:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                else:
                    silence_duration = current_time - self.silence_start_time
                    if silence_duration >= self.SILENCE_THRESHOLD:
                        await self.manager.send_json(self.websocket, {"type": "pause", "message": "检测到语音，暂停前端录音"})
                        logger.info("检测到沉默，结束录音段落。")
                        return await self._save_audio_file()
                self.audio_data.append(audio_chunk)
            return None

    async def _save_audio_file(self):
        if not self.audio_data or not self.is_recording:
            self.reset()
            return None
        
        chunk_duration = self.CHUNK / self.SAMPLE_RATE
        num_silence_chunks = int(self.SILENCE_THRESHOLD / chunk_duration)
        if len(self.audio_data) > num_silence_chunks:
            audio_data = self.audio_data[:-num_silence_chunks]
        else:
            audio_data = []

        self.reset()

        if not audio_data:
            logger.info("录音数据不足，忽略本段。")
            await self.manager.send_json(self.websocket, {"type": "resume", "message": "未检测到有效语音，请继续说话"})
            return None

        try:
            audio_array = np.concatenate([np.frombuffer(chunk, np.int16) for chunk in audio_data])
            from datetime import datetime
            output_file = os.path.join(self.temp_dir, f"recorded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            import soundfile as sf
            sf.write(output_file, audio_array, self.SAMPLE_RATE)
            logger.info(f"音频已保存到: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"保存音频出错: {e}")
            await self.manager.send_json(self.websocket, {"type": "error", "message": f"保存音频出错: {e}"})
            return None

    def reset(self):
        self.is_recording = False
        self.silence_start_time = None
        self.audio_data = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket 已连接")

    temp_dir = tempfile.mkdtemp()
    audio_processor = WebSocketAudioProcessor(websocket, manager, vad_model, temp_dir)

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                if not data:
                    break

                audio_data = np.frombuffer(data, dtype=np.float32)
                audio_data = (audio_data * 32768).astype(np.int16)

                audio_file = await audio_processor.process_audio_chunk(audio_data)

                if audio_file:
                    try:
                        text = Audio2Text.get_text(asr_model, audio_file)
                        logger.info(f"识别文本: {text}")
                        await manager.send_json(websocket, {"type": "text", "message": text})


                        if '上一步' in text:
                            response='好的，已跳转到上一步'
                            await manager.send_json(websocket, {"type": "1", "message": response})
                        elif '下一步' in text:
                            response='好的，已跳转到下一步'
                            await manager.send_json(websocket, {"type": "2", "message": response})
                        else :

                            logger.info("生成模型回答")
                            response = llm.llm.invoke(text)
                            print(response)
                            wav, sr, spec = tts.infer(
        ref_file='C:\\Users\陈飞扬\Desktop\\fronted\\fronted\\luyin.wav',
        ref_text="""你好，我是智小厨，你的智能烹饪助手。""",
        gen_text=response,
        file_wave=None,
        seed=None
                            )
                            # import sounddevice as sd  
                            # def play_audio_from_array(wav, sr):
                            #     """直接播放 NumPy 数组形式的音频数据"""
                            #     try:
                            #         sd.play(wav, sr)  # 播放音频
                            #         sd.wait()  # 等待播放完成
                            #     except Exception as e:
                            #         print(f"播放音频时出错: {e}")
                            # play_audio_from_array(wav, sr)
                            import io
                            import soundfile as sf
                            buf = io.BytesIO()
                            sf.write(buf, wav, sr, format='WAV')    
                            audio_bytes = buf.getvalue()
            # 发送音频流
            # await websocket.send_bytes(audio_bytes)
            # 也可以加上类型标识，前端更好识别
                            await websocket.send_json({"type": "audio", "data": base64.b64encode(audio_bytes).decode()})

                            await manager.send_json(websocket, {"type": "response", "message": response})

                            # await manager.send_json(websocket, {"type": "response", "message": response})

                    finally:
                        if os.path.exists(audio_file):
                            os.remove(audio_file)

                        await manager.send_json(websocket, {"type": "resume", "message": "处理完成，请继续说话"})

            except Exception as e:
                logger.error(f"处理音频出错: {e}")
                await manager.send_json(websocket, {"type": "error", "message": str(e)})
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        if websocket in manager.active_connections:
            await manager.send_json(websocket, {"type": "error", "message": str(e)})
            manager.disconnect(websocket)
    finally:
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        logger.info("连接清理完成")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
