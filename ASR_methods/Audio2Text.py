from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import ASR_methods.Get_AudioFile as Get_AudioFile

model_dir = 'models\FunAudioLLM\SenseVoiceSmall'

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

def get_text(chunk,model=model):
    a=Get_AudioFile.VAD_audio()
    filename= a.get_audio(chunk=chunk)
    res = model.generate(
        input= filename,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)
    return text
