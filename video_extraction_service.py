import os
import json
import subprocess
from pathlib import Path

import whisperx
import whisperx.diarize

# ------------------------- CONFIG -------------------------

CHANNEL_URL = "https://youtube.com/@alratv"
AUDIO_DIR = Path("audio_downloads")
OUTPUT_DIR = Path("extraction_output")
DEVICE = "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL_SIZE = "small"
LANGUAGE = "ur"
HF_TOKEN = os.environ.get("HF_TOKEN")

MAX_VIDEOS = 4
MANUAL_VIDEO_IDS = [
    "wl9Oej-yXPU",
    "BgRczEbeuvE",
    "YCzrkM0rgCg",
    "tl_wSJ4zfHY",
]

AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
# ------------------------- STEP 1: LIST VIDEOS -------------------------

def get_channel_video_ids(channel_url: str) -> list[dict]:
    """Return a flat list of {video_id, url} for every video on the channel,
    without downloading anything yet."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    video_ids = [vid.strip() for vid in result.stdout.splitlines() if vid.strip()]
    return [{"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}"} for vid in video_ids]


def already_processed(video_id: str) -> bool:
    """Skip videos we've already run through this pipeline."""
    return (OUTPUT_DIR / f"{video_id}.json").exists()
# ------------------------- STEP 2: DOWNLOAD AUDIO -------------------------

def download_audio(video_url: str, video_id: str) -> Path:
    """Download audio-only, convert to 16kHz mono WAV (what WhisperX expects)."""
    out_path = AUDIO_DIR / f"{video_id}.wav"
    if out_path.exists():
        return out_path

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", str(AUDIO_DIR / f"{video_id}.%(ext)s"),
        video_url,
    ]
    subprocess.run(cmd, check=True)
    return out_path
def extract_audio_from_local_file(video_path: str, video_id: str) -> Path:
    """Extract and convert audio from a local video file (not from YouTube)."""
    out_path = AUDIO_DIR / f"{video_id}.wav"
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
# ------------------------- STEP 3-6: TRANSCRIBE + ALIGN + DIARIZE -------------------------

def transcribe_and_diarize(audio_path: Path) -> dict:
    """Runs WhisperX transcription, alignment, and speaker diarization.
    Returns dict with num_speakers and speaker-labeled segments."""

    # 3. Transcribe
    model = whisperx.load_model(WHISPER_MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=LANGUAGE, batch_size=16)

    # 4. Align for word-level timestamps
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False)

    # 5. Diarize (speaker detection)
    diarize_model = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio)

    # 6. Merge speaker labels into transcript
    result = whisperx.assign_word_speakers(diarize_segments, result)

    speakers = {seg.get("speaker") for seg in result["segments"] if seg.get("speaker")}

    return {
        "num_speakers": len(speakers),
        "speakers": sorted(speakers),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ],
    }
# ------------------------- STEP 7: SAVE OUTPUT -------------------------

def save_output(video_id: str, transcript_data: dict):
    output = {
        "video_id": video_id,
        "num_speakers": transcript_data["num_speakers"],
        "speaker_labels": transcript_data["speakers"],
        "script": transcript_data["segments"],
    }
    out_path = OUTPUT_DIR / f"{video_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


# ------------------------- MAIN PIPELINE -------------------------
def run():
    print("Fetching video list from channel...")
    if MANUAL_VIDEO_IDS:
        videos = [
            {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}"}
            for vid in MANUAL_VIDEO_IDS
        ]
        print(f"Using manual list of {len(videos)} video(s).")
    else:
        videos = get_channel_video_ids(CHANNEL_URL)
        print(f"Found {len(videos)} videos.")

    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]

    for video in videos:
        vid = video["video_id"]
        if already_processed(vid):
            print(f"Skipping {vid} (already processed)")
            continue

        print(f"Processing {vid}...")
        audio_path = download_audio(video["url"], vid)
        transcript_data = transcribe_and_diarize(audio_path)

        if transcript_data["num_speakers"] > 1:
            print(f"Discarded {vid}: {transcript_data['num_speakers']} speakers detected (only single-speaker audio is kept).")
            continue

        save_output(vid, transcript_data)


def test_local_file(video_path: str, video_id: str):
    """Quick test: run diarization on a local video file and print the speaker count."""
    print(f"Testing local file: {video_path}")
    audio_path = extract_audio_from_local_file(video_path, video_id)
    transcript_data = transcribe_and_diarize(audio_path)

    print(f"Detected {transcript_data['num_speakers']} speaker(s): {transcript_data['speakers']}")

    if transcript_data["num_speakers"] > 1:
        print(f"DISCARD RULE TRIGGERED: {transcript_data['num_speakers']} speakers detected — this audio would be discarded.")
    else:
        print("Single speaker (or silence) — this audio would be kept.")

    save_output(video_id, transcript_data)

if __name__ == "__main__":
    # Uncomment ONE of the lines below to choose what to run:

    run()  # process the MANUAL_VIDEO_IDS / channel list as usual

    # test_local_file(r"C:\Users\siyaa\Downloads\test1.mp4", "test1_local")