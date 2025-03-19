# -*- encoding: utf-8 -*-
import hashlib
import hmac
import base64
import json
import time
import threading
import re
from websocket import create_connection
from urllib.parse import quote
import logging
import pyaudio
import requests
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import websocket
import datetime
from wsgiref.handlers import format_date_time
from time import mktime
import _thread as thread
import os
from urllib.parse import urlencode
import ssl
# 实时语音转写配置  
RTASR_APP_ID = "a4ab676a"  
RTASR_API_KEY = "c0a8020c200794a61aaa108740ab170c" 

# 语音合成配置
TTS_APP_ID = '3b2f8d66'
TTS_API_SECRET = 'NWE0MjY1N2Y0NGU2NDUwNTQ3NzhkN2Qx'
TTS_API_KEY = '5fe3fa9b6a931c920448c48ed534543d'

# 沉默时间阈值（秒）
SILENCE_THRESHOLD = 3  

class TextToSpeech:
    STATUS_FIRST_FRAME = 0
    STATUS_CONTINUE_FRAME = 1
    STATUS_LAST_FRAME = 2

    def __init__(self, appid=TTS_APP_ID, api_key=TTS_API_KEY, api_secret=TTS_API_SECRET):
        self.APPID = appid
        self.APIKey = api_key
        self.APISecret = api_secret
        self.audio_data = bytearray()

    def _create_url(self, text):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: ws-api.xfyun.cn\n" + "date: " + date + "\n" + "GET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode('utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + '?' + urlencode(v)

    def _on_message(self, ws, message):
        try:
            message = json.loads(message)
            code = message["code"]
            audio = base64.b64decode(message["data"]["audio"])
            status = message["data"]["status"]
            if code != 0:
                print(f"TTS error: {message['message']} code: {code}")
            else:
                self.audio_data.extend(audio)
            if status == self.STATUS_LAST_FRAME:
                ws.close()
        except Exception as e:
            print(f"TTS message parse exception: {e}")

    def _on_error(self, ws, error):
        print(f"TTS error: {error}")

    def _on_close(self, ws, *args, **kwargs):
        print("TTS WebSocket closed")

    def _on_open(self, ws):
        def run(*args):
            common_args = {"app_id": self.APPID}
            business_args = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
            data_args = {"status": self.STATUS_LAST_FRAME, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
            d = {"common": common_args, "business": business_args, "data": data_args}
            ws.send(json.dumps(d))
        thread.start_new_thread(run, ())

    def synthesize(self, text):
        self.Text = text
        self.audio_data = bytearray()
        ws_url = self._create_url(text)
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(ws_url, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
        ws.on_open = self._on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return bytes(self.audio_data)

class Client:
    def __init__(self):
        # 初始化实时语音转写
        self.rtasr_ws = None
        self.setup_rtasr()

        # 初始化语音合成
        self.tts = TextToSpeech()

        # 用于存储转录文本和时间戳
        self.text_buffer = []
        self.last_text_time = time.time()
        
        # 初始化大模型
        try:
            self.llm = Ollama(model="qwen2.5:7b-instruct", temperature=0.5)
            self.memory = ConversationBufferMemory()
            self.conversation = ConversationChain(llm=self.llm, memory=self.memory)
            print("本地大模型初始化成功")
        except Exception as e:
            print(f"大模型初始化失败: {e}")
            self.llm = None

        # 初始化音频播放
        self.audio_player = pyaudio.PyAudio()

        # 启动接收线程
        self.trecv_rtasr = threading.Thread(target=self.recv_rtasr)
        self.trecv_rtasr.daemon = True
        self.trecv_rtasr.start()

        # 启动定时检查线程
        self.tcheck = threading.Thread(target=self.check_silence)
        self.tcheck.daemon = True
        self.tcheck.start()

    def setup_rtasr(self):
        base_url = "ws://rtasr.xfyun.cn/v1/ws"
        ts = str(int(time.time()))
        tt = (RTASR_APP_ID + ts).encode('utf-8')
        md5 = hashlib.md5()
        md5.update(tt)
        base_string = md5.hexdigest().encode('utf-8')
        signa = hmac.new(RTASR_API_KEY.encode('utf-8'), base_string, hashlib.sha1).digest()
        signa = base64.b64encode(signa).decode('utf-8')
        self.rtasr_url = f"{base_url}?appid={RTASR_APP_ID}&ts={ts}&signa={quote(signa)}"
        self.end_tag = '{"end": true}'
        try:
            self.rtasr_ws = create_connection(self.rtasr_url)
            print(f"实时语音转写连接成功: {self.rtasr_url}")
        except Exception as e:
            print(f"实时语音转写连接失败: {e}")

    def send_audio(self):
        if not self.rtasr_ws:
            print("语音转写未连接，无法发送")
            return
        CHUNK = 1280
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("开始录音，按 Ctrl+C 停止...")
        try:
            while True:
                audio_data = stream.read(CHUNK)
                self.rtasr_ws.send(audio_data)
                time.sleep(0.04)
        except KeyboardInterrupt:
            self.rtasr_ws.send(self.end_tag.encode('utf-8'))
            print("语音转写发送结束标志成功")
            self.process_buffered_text()
        except Exception as e:
            print(f"录音或发送失败: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.close()

    def recv_rtasr(self):
        if not self.rtasr_ws:
            return
        try:
            while self.rtasr_ws.connected:
                result = self.rtasr_ws.recv()
                if not result:
                    print("语音转写接收结果结束")
                    break
                result_dict = json.loads(result)
                if result_dict["action"] == "started":
                    print("语音转写握手成功: " + result)
                elif result_dict["action"] == "result":
                    try:
                        data = json.loads(result_dict["data"])
                        text = ""
                        if "cn" in data and "st" in data["cn"] and "rt" in data["cn"]["st"]:
                            for rt in data["cn"]["st"]["rt"]:
                                for ws in rt["ws"]:
                                    for cw in ws["cw"]:
                                        text += cw["w"]
                            print("实时转录结果: " + text)
                            self.update_text_buffer(text)
                            self.last_text_time = time.time()
                    except Exception as e:
                        print(f"解析转录结果失败: {e}")
                elif result_dict["action"] == "error":
                    print("语音转写错误: " + result)
                    self.rtasr_ws.close()
                    return
        except Exception as e:
            print(f"语音转写接收异常: {e}")

    def update_text_buffer(self, new_text):
        if not self.text_buffer:
            self.text_buffer.append(new_text)
            return
        last_text = self.text_buffer[-1]
        if new_text.startswith(last_text):
            self.text_buffer[-1] = new_text
        elif self.is_similar(last_text, new_text):
            merged_text = self.merge_texts(last_text, new_text)
            self.text_buffer[-1] = merged_text
        elif new_text.startswith(("。", "？", "！", "，")):
            if len(new_text) <= 2:
                if not last_text.endswith(("。", "？", "！")):
                    self.text_buffer[-1] = last_text + new_text
            else:
                self.text_buffer.append(new_text)
        else:
            self.text_buffer.append(new_text)

    def is_similar(self, text1, text2):
        if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.7:
            return False
        min_length = min(len(text1), len(text2))
        for i in range(min(min_length, 5), 0, -1):
            for j in range(len(text1) - i + 1):
                substring = text1[j:j+i]
                if substring in text2:
                    return True
        return False

    def merge_texts(self, text1, text2):
        if text1 in text2:
            return text2
        if text2 in text1:
            return text1
        common = self.longest_common_substring(text1, text2)
        if not common:
            return text1 + text2
        pos1 = text1.find(common)
        pos2 = text2.find(common)
        prefix = text1[:pos1] if pos1 < pos2 else text2[:pos2]
        suffix = text1[pos1+len(common):] if pos1+len(common) < len(text1) else text2[pos2+len(common):]
        return prefix + common + suffix

    def longest_common_substring(self, s1, s2):
        if not s1 or not s2:
            return ""
        m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        longest, end_pos = 0, 0
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i-1] == s2[j-1]:
                    m[i][j] = m[i-1][j-1] + 1
                    if m[i][j] > longest:
                        longest = m[i][j]
                        end_pos = i
        return s1[end_pos - longest:end_pos]

    def check_silence(self):
        while True:
            try:
                current_time = time.time()
                if self.text_buffer and (current_time - self.last_text_time >= SILENCE_THRESHOLD):
                    self.process_buffered_text()
                time.sleep(1)
            except Exception as e:
                print(f"检查沉默时间异常: {e}")
                time.sleep(1)

    def process_buffered_text(self):
        if not self.text_buffer:
            return
        try:
            final_text = self.smart_join_texts(self.text_buffer)
            final_text = self.clean_text(final_text)
            print(f"整合后的完整文本: {final_text}")
            if final_text.strip():
                self.nlp_process(final_text)
            self.text_buffer = []
        except Exception as e:
            print(f"处理缓冲文本异常: {e}")

    def smart_join_texts(self, texts):
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        result = texts[0]
        for i in range(1, len(texts)):
            current = texts[i]
            if current in result:
                continue
            overlap = False
            min_overlap_len = min(3, min(len(result), len(current)))
            for j in range(min(len(result), 10), min_overlap_len - 1, -1):
                if result[-j:] == current[:j]:
                    result += current[j:]
                    overlap = True
                    break
            if not overlap:
                if result.endswith(("。", "？", "！", ".", "?", "!")):
                    result += " " + current
                else:
                    result += "，" + current
        return result

    def clean_text(self, text):
        if not text:
            return ""
        for punct in ["。", "，", "？", "！", "、", "；", "：", ".", ",", "?", "!", ";", ":"]:
            while punct + punct in text:
                text = text.replace(punct + punct, punct)
        text = re.sub(r'([一-龥])\1+', r'\1', text)
        for length in range(2, 6):
            for i in range(len(text) - length * 2 + 1):
                chunk = text[i:i+length]
                if chunk == text[i+length:i+length*2]:
                    text = text[:i+length] + text[i+length*2:]
                    i = max(0, i-length)
        filler_words = ["嗯", "啊", "呃", "那个", "这个", "就是", "然后", "所以", "其实", "毕竟", "你看", "怎么说"]
        for word in filler_words:
            if text.count(word) > 2:
                indices = [i for i, _ in enumerate(text) if text[i:i+len(word)] == word]
                for idx in indices[1:-1]:
                    text = text[:idx] + text[idx+len(word):]
        text = re.sub(r'(嗯|啊|呃)\1+', r'\1', text)
        text = re.sub(r'([。？！])[，,]', r'\1', text)
        return text

    def play_audio(self, audio_data, rate=16000):
        """播放音频数据"""
        stream = self.audio_player.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def nlp_process(self, text):
        if not text.strip():
            return
        print(f"正在处理文本: {text}")
        try:
            if self.llm:
                print("AI回复: ", end="", flush=True)
                response = ""
                for chunk in self.llm.stream(text):
                    print(chunk, end="", flush=True)
                    response += chunk
                print("\n")
                # 语音合成并播放
                audio_data = self.tts.synthesize(response)
                self.play_audio(audio_data)
            else:
                print("大模型未初始化，无法处理文本")
        except Exception as e:
            print(f"NLP处理异常: {e}")

    def close(self):
        if self.rtasr_ws and self.rtasr_ws.connected:
            self.rtasr_ws.close()
            print("实时语音转写连接已关闭")
        self.audio_player.terminate()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = Client()
    client.send_audio()