# Author: Mynk
"""Optional microphone/speech check. Audio is sent to Google's speech API only when run."""

from pathlib import Path
import re

import nltk
import speech_recognition as sr


ROOT = Path(__file__).resolve().parent
PAPER_PATH = ROOT / "paper.txt"
TRANSCRIPT_PATH = ROOT / "test.txt"


def _normalised_words(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import wordpunct_tokenize

    ignored = set(stopwords.words("english"))
    return [word.lower() for word in wordpunct_tokenize(text) if re.search(r"[a-z0-9]", word, re.I) and word.lower() not in ignored]


def run_audio_check(record_seconds=30):
    """Record once, transcribe with Google Speech Recognition, and compare with paper.txt."""
    if not PAPER_PATH.is_file():
        print("Audio check skipped: create paper.txt in the project folder with the exam question text first.")
        return 2

    try:
        if not nltk.data.find("corpora/stopwords"):
            raise LookupError
    except LookupError:
        print("Downloading the NLTK English stopword list for the first run...")
        if not nltk.download("stopwords", quiet=False):
            print("Could not download NLTK stopwords; audio check cannot continue.")
            return 2

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting microphone for room noise; please stay quiet briefly...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"Recording for up to {record_seconds} seconds. Speak normally.")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=record_seconds)
    except sr.WaitTimeoutError:
        print("No speech detected before the recording timed out.")
        return 2
    except (OSError, AttributeError) as exc:
        print(f"Could not open the microphone: {exc}")
        return 2

    try:
        # This call transmits the recording to Google's Speech Recognition service.
        transcript = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Speech was not clear enough to transcribe.")
        return 2
    except sr.RequestError as exc:
        print(f"Google Speech Recognition request failed: {exc}")
        return 2

    TRANSCRIPT_PATH.write_text(transcript + "\n", encoding="utf-8")
    paper = PAPER_PATH.read_text(encoding="utf-8", errors="replace")
    spoken_words = _normalised_words(transcript)
    paper_words = set(_normalised_words(paper))
    matches = sorted(set(spoken_words) & paper_words)

    print(f"Transcript saved locally to {TRANSCRIPT_PATH.name}.")
    print(f"Question-paper word matches: {len(matches)}")
    print("Matching terms:", ", ".join(matches) if matches else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_audio_check())
