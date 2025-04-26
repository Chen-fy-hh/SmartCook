# 调用ASR模型进行语音识别
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr import AutoModel

def ASR_Model_init():
    
    model_dir = 'models\FunAudioLLM\SenseVoiceSmall'
    model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    )   
    print("ASR模型加载成功")
    return model


def get_text(asr_model,filename,language="auto"):
   
    res = asr_model.generate(
        input= filename,
        cache={},
        language=language,  # "zn", "en", "yue", "ja", "ko"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    return text
