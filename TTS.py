import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from time import mktime
import _thread as thread
import os

class TextToSpeech:
    STATUS_FIRST_FRAME = 0  # 第一帧的标识
    STATUS_CONTINUE_FRAME = 1  # 中间帧标识
    STATUS_LAST_FRAME = 2  # 最后一帧的标识

    def __init__(self, appid='3b2f8d66', api_key='5fe3fa9b6a931c920448c48ed534543d', 
                 api_secret='NWE0MjY1N2Y0NGU2NDUwNTQ3NzhkN2Qx'):
        """初始化类，设置API参数"""
        self.APPID = appid
        self.APIKey = api_key
        self.APISecret = api_secret
        self.audio_data = bytearray()  # 用于存储生成的音频流

    def _create_url(self, text):
        """生成WebSocket鉴权URL"""
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + '?' + urlencode(v)

    def _on_message(self, ws, message):
        """处理接收到的消息"""
        try:
            message = json.loads(message)
            code = message["code"]
            sid = message["sid"]
            audio = message["data"]["audio"]
            audio = base64.b64decode(audio)
            status = message["data"]["status"]

            if code != 0:
                print(f"sid:{sid} call error:{message['message']} code is:{code}")
            else:
                self.audio_data.extend(audio)  # 将音频数据追加到缓冲区

            if status == self.STATUS_LAST_FRAME:
                print("Synthesis completed, WebSocket is closed")
                ws.close()

        except Exception as e:
            print(f"Receive message, but parse exception: {e}")

    def _on_error(self, ws, error):
        """处理WebSocket错误"""
        print(f"### error: {error}")

    def _on_close(self, ws, *args, **kwargs):
        """处理WebSocket关闭"""
        print("### closed ###")

    def _on_open(self, ws):
        """处理WebSocket连接建立，发送数据"""
        def run(*args):
            # 公共参数和业务参数
            common_args = {"app_id": self.APPID}
            business_args = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": "xiaoyan", "tte": "utf8"}
            data_args = {"status": self.STATUS_LAST_FRAME, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}

            d = {"common": common_args, "business": business_args, "data": data_args}
            ws.send(json.dumps(d))
            print("------> 开始发送文本数据")

        thread.start_new_thread(run, ())

    def synthesize(self, text, output_file=None):
        """输入文本，生成语音流，可选择保存到文件，返回音频数据"""
        self.Text = text
        self.audio_data = bytearray()  # 重置音频数据

        # 创建WebSocket连接
        ws_url = self._create_url(text)
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(ws_url, 
                                   on_message=self._on_message, 
                                   on_error=self._on_error, 
                                   on_close=self._on_close)
        ws.on_open = self._on_open

        # 运行WebSocket，直到完成
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        # 如果指定了输出文件，则保存
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(self.audio_data)

        return bytes(self.audio_data)  # 返回音频流
    