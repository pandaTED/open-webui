from transformers import pipeline
import time


begin_1 = time.time()
transcriber = pipeline(
    "automatic-speech-recognition",
    model="BELLE-2/Belle-whisper-large-v3-turbo-zh",
    device="cuda",
)
transcriber.model.config.forced_decoder_ids = (
    transcriber.tokenizer.get_decoder_prompt_ids(
        language="zh",
        task="transcribe"
    )
)
transcription = transcriber("data/cache/audio/transcriptions/6fd4ee35-7a54-4b0c-9026-d68ed8bdedae.wav")

end_1 = time.time()
duration = end_1 - begin_1

print(f"识别花费： {duration}")

print(transcription)
