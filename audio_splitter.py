from pydub import AudioSegment
import os


def split_audio_into_chunks(file_path, chunk_length_ms=2000):
    # 1. 파일 불러오기
    print(f"파일을 불러오는 중...: {file_path}")
    audio = AudioSegment.from_file(file_path)

    # 2. 저장할 폴더 생성 (파일명과 동일한 이름)
    base_name = os.path.basename(file_path).split('.')[0]
    output_dir = f"./output_{base_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 자르기 루프
    duration_ms = len(audio)
    print(f"전체 길이: {duration_ms / 1000}초")

    count = 0
    for start_ms in range(0, duration_ms, chunk_length_ms):
        end_ms = start_ms + chunk_length_ms

        # 마지막 조각이 2초보다 짧을 경우 제외하거나 포함할 수 있음
        if end_ms > duration_ms:
            break

        chunk = audio[start_ms:end_ms]

        # 4. 파일 저장 (예: sound_0001.wav)
        file_name = f"{base_name}_{count:04d}.wav"
        chunk.export(os.path.join(output_dir, file_name), format="wav")

        count += 1
        if count % 100 == 0:
            print(f"{count}개 파일 생성 완료...")

    print(f"✅ 작업 완료! 총 {count}개의 파일이 '{output_dir}' 폴더에 저장되었습니다.")


# 실행 예시
split_audio_into_chunks("your_audio_file.wav")