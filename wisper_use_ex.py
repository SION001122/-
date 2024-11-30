import whisper

# Whisper 모델 로드
model = whisper.load_model("base")  # "tiny", "base", "small", "medium", "large" 중 선택 가능

# 음성 파일 변환
result = model.transcribe("B.wav")  # "B.wav"는 변환할 파일 이름

# 결과 출력
print("Text:", result["text"])
