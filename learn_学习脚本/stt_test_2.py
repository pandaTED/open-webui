import time

from faster_whisper import WhisperModel


begin_1 = time.time()

model = WhisperModel("data/cache/whisper/models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478")
segments, info = model.transcribe("data/cache/audio/transcriptions/6fd4ee35-7a54-4b0c-9026-d68ed8bdedae.wav")

end_1 = time.time()
duration = end_1 - begin_1

print(f"识别花费： {duration}")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

