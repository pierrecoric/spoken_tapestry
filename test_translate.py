import whisper
from phonemizer import phonemize
import csv

model = whisper.load_model("base")
audio_analyze = model.transcribe("test.wav", word_timestamps=True)

segments = audio_analyze["segments"]
words_with_timestamps = []

for segment in segments:
    for word_info in segment["words"]:
        word = word_info["word"].strip(".,?! ")
        start = word_info["start"]
        end = word_info["end"]
        words_with_timestamps.append((word, start, end))

words = [word for word, _, _ in words_with_timestamps]
ipa_transcriptions = phonemize(words, language='en-us', backend='espeak', strip=True, njobs=1)

rows = list(zip(words, ipa_transcriptions, [start for _, start, _ in words_with_timestamps], [end for _, _, end in words_with_timestamps]))

with open("ipa_words_timestamps.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["word", "ipa", "start_timestamp", "end_timestamp"])
    writer.writerows(rows)

print("CSV written: ipa_words_timestamps.csv")
