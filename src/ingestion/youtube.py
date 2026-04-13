"""Fetch raw transcripts and metadata from YouTube."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests.exceptions
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from a YouTube URL or return the ID as-is.

    Supported URL formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - bare 11-character video ID

    Args:
        url_or_id: A full YouTube URL or a bare video ID string.

    Returns:
        The 11-character YouTube video ID.

    Raises:
        ValueError: If no valid video ID can be parsed from the input.
    """
    url_or_id = url_or_id.strip()

    if "youtube.com/watch" in url_or_id:
        parsed = urlparse(url_or_id)
        ids = parse_qs(parsed.query).get("v", [])
        if not ids:
            raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id!r}")
        return ids[0]

    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[-1].split("?")[0].split("/")[0]

    if "youtube.com/embed/" in url_or_id:
        return url_or_id.split("youtube.com/embed/")[-1].split("?")[0].split("/")[0]

    # Accept a bare 11-character video ID (alphanumeric + _ + -)
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", url_or_id):
        return url_or_id

    raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id!r}")


def fetch_transcript(url: str) -> list[dict[str, Any]]:
    """Fetch the transcript for a YouTube video given its URL or video ID.

    Each entry in the returned list has keys:
        - text (str): the spoken words for that segment
        - start_time (float): segment start time in seconds
        - duration (float): length of the segment in seconds

    Args:
        url: A full YouTube URL in any supported format, or a bare video ID.

    Returns:
        List of transcript segment dicts.

    Raises:
        ValueError: If the URL is invalid or the transcript is empty.
        VideoUnavailable: If the video is private or does not exist.
        TranscriptsDisabled: If captions have been disabled for the video.
        NoTranscriptFound: If no transcript (auto-generated or manual) is available.
        requests.exceptions.RequestException: On network-level failures.
    """
    try:
        video_id = extract_video_id(url)
    except ValueError as exc:
        raise ValueError(f"Invalid YouTube URL: {url!r}") from exc

    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        # Prefer English; fall back to any available language (auto-generated or manual).
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
    except VideoUnavailable as exc:
        raise RuntimeError(
            f"Video '{video_id}' is unavailable or private."
        ) from exc
    except TranscriptsDisabled as exc:
        raise RuntimeError(
            f"Transcripts are disabled for video '{video_id}'."
        ) from exc
    except NoTranscriptFound as exc:
        raise RuntimeError(
            f"No transcript found for video '{video_id}'. It may lack captions."
        ) from exc
    except (YouTubeRequestFailed, requests.exceptions.RequestException) as exc:
        raise RuntimeError(f"Network error fetching transcript: {exc}") from exc

    segments = list(fetched)
    if not segments:
        raise ValueError(f"Empty transcript returned for video '{video_id}'.")

    return [
        {
            "text": seg.text,
            "start_time": float(seg.start),
            "duration": float(seg.duration),
        }
        for seg in segments
    ]


def get_video_title(video_id: str) -> str:
    """Retrieve the title of a YouTube video using yt-dlp.

    Args:
        video_id: The 11-character YouTube video ID.

    Returns:
        The video title string, or 'Unknown Title' if it cannot be fetched.
    """
    try:
        import yt_dlp  # imported lazily to avoid hard dependency at module load

        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown Title") or "Unknown Title"
    except Exception:  # noqa: BLE001
        return "Unknown Title"
