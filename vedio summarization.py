
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Minimal imports for runtime; we import heavy libs lazily where possible

@dataclass
class SummarizationConfig:
    whisper_model: str = "whisper-1"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    chunk_size: int = 4000  # characters per chunk for splitting transcript
    chunk_overlap: int = 400
    max_concurrency: int = 4  # LangChain parallelism hint


def ensure_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
        )
    return key


def extract_audio_to_wav(video_path: Path) -> Path:
    """Extract audio from the given video file into a temporary WAV file and return its path."""
    try:
        from moviepy.editor import VideoFileClip  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "moviepy is required for audio extraction. Please install dependencies (see requirements)."
        ) from e

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create a temporary WAV file
    tmp_dir = tempfile.mkdtemp(prefix="video_summarizer_")
    wav_path = Path(tmp_dir) / (video_path.stem + ".wav")

    try:
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                raise RuntimeError("No audio track found in the video.")
            # 16k mono WAV is typically good for ASR APIs
            clip.audio.write_audiofile(
                str(wav_path),
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],  # mono
                verbose=False,
                logger=None,
            )
    except OSError as e:
        # Often happens when ffmpeg is missing
        raise RuntimeError(
            "FFmpeg error while extracting audio. Ensure ffmpeg is installed and on PATH."
        ) from e

    return wav_path


def transcribe_audio_with_whisper(audio_path: Path, model: str = "whisper-1", prompt: Optional[str] = None) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    ensure_openai_key()
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "The 'openai' package is required. Please install project dependencies."
        ) from e

    client = OpenAI()
    # Open file in binary mode for upload
    with open(audio_path, "rb") as f:
        try:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=f,
                prompt=prompt or None,
                response_format="text",  # returns plain text transcript
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenAI Whisper API transcription failed: {e}"
            ) from e

    # The v1 SDK returns a str if response_format="text"
    if isinstance(transcription, str):
        return transcription

    # Fallback in case SDK returns object
    text = getattr(transcription, "text", None)
    if not text:
        raise RuntimeError("Failed to retrieve transcription text from OpenAI response.")
    return text


def summarize_transcript_with_langchain(transcript: str, cfg: SummarizationConfig, style_prompt: Optional[str] = None) -> str:
    """Summarize long transcripts via LangChain map-reduce chain using OpenAI chat model."""
    ensure_openai_key()
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
        from langchain.docstore.document import Document  # type: ignore
        from langchain.chains.summarize import load_summarize_chain  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LangChain and langchain-openai are required. Please install project dependencies."
        ) from e

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    docs: List[Document] = [Document(page_content=chunk) for chunk in text_splitter.split_text(transcript)]
    if not docs:
        return ""

    llm = ChatOpenAI(model=cfg.llm_model, temperature=cfg.temperature)

    default_style = (
        "Provide a concise, well-structured summary of the video transcript. "
        "Include key points, important facts, and any action items. "
        "Use bullet points when appropriate and keep it factual and neutral."
    )
    style = style_prompt or default_style

    # Map-reduce chain: summarize chunks, then combine
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=None,
        combine_prompt=None,
        verbose=False,
    )

    # Inject style instruction by prepending to each chunk
    styled_docs = [
        Document(page_content=f"Instruction: {style}\n\nTranscript chunk:\n{d.page_content}")
        for d in docs
    ]

    summary = chain.run(styled_docs)
    return summary.strip()


def summarize_video(
    video_path: str,
    *,
    whisper_model: str = "whisper-1",
    llm_model: str = "gpt-4o-mini",
    style_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """High-level convenience function: extract -> transcribe -> summarize -> return summary text."""
    cfg = SummarizationConfig(
        whisper_model=whisper_model,
        llm_model=llm_model,
        temperature=temperature,
    )

    video = Path(video_path)
    audio_path = extract_audio_to_wav(video)
    try:
        transcript = transcribe_audio_with_whisper(audio_path, model=cfg.whisper_model)
    finally:
        # Clean temporary audio file if it exists
        try:
            if audio_path.exists():
                audio_path.unlink()
                # Also remove temp directory if empty
                parent = audio_path.parent
                if parent.exists():
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
        except Exception:
            pass

    summary = summarize_transcript_with_langchain(transcript, cfg, style_prompt=style_prompt)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a video's content using OpenAI Whisper + LLM")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument("--output", help="Path to save summary text (optional)")
    parser.add_argument(
        "--prompt",
        help="Optional style or instruction prompt to guide the summary",
        default=None,
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="OpenAI chat model to use for summarization (e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--whisper-model",
        default="whisper-1",
        help="OpenAI Whisper model for transcription",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="LLM temperature for summarization"
    )
    parser.add_argument(
        "--save-transcript",
        dest="save_transcript",
        help="Optional path to save full transcript before summarizing",
        default=None,
    )

    args = parser.parse_args(argv)

    # Basic validation
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 2

    try:
        # Extract audio, transcribe, summarize
        audio_path = extract_audio_to_wav(video_path)
        try:
            transcript = transcribe_audio_with_whisper(audio_path, model=args.whisper_model)
        finally:
            try:
                if audio_path.exists():
                    audio_path.unlink()
                    parent = audio_path.parent
                    if parent.exists():
                        try:
                            parent.rmdir()
                        except OSError:
                            pass
            except Exception:
                pass

        if args.save_transcript:
            Path(args.save_transcript).write_text(transcript, encoding="utf-8")

        cfg = SummarizationConfig(
            whisper_model=args.whisper_model,
            llm_model=args.llm_model,
            temperature=args.temperature,
        )
        summary = summarize_transcript_with_langchain(transcript, cfg, style_prompt=args.prompt)

    except Exception as e:
        print(f"Failed to summarize video: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(summary, encoding="utf-8")
        print(f"Summary saved to: {args.output}")
    else:
        print("\n===== SUMMARY =====\n")
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
