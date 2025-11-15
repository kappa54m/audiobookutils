from dataclasses import dataclass

import os
from typing import Iterable
from pathlib import Path


@dataclass
class TextSegment:
    """
    Segment of text (e.g., chapter)
    """
    name: str|None
    text: str

    def to_serializable_dict(self):
        return {
            'name': str(self.name) if self.name is not None else None,
            'text': str(self.text)
        }


def textsegment_from_dict(d: dict) -> TextSegment:
    return TextSegment(name=d['name'], text=d['text'])


@dataclass
class Ebook:
    title: str
    author: str
    chapters: list[TextSegment]

    def to_serializable_dict(self):
        return {
            'title': str(self.title),
            'author': str(self.author),
            'chapters': [ts.to_serializable_dict() for ts in self.chapters]
        }


def ebook_from_dict(d: dict) -> Ebook:
    return Ebook(title=d['title'], author=d['author'],
                 chapters=[textsegment_from_dict(ts) for ts in d['chapters']])


@dataclass(eq=True, frozen=True)
class CorpusLocation:
    """
    Location of a character within a corpus
    """
    document_index: int
    character_index: int

    def to_serializable_dict(self):
        return {
            'document_index': self.document_index,
            'character_index': self.character_index
        }


def corpuslocation_from_dict(d: dict) -> CorpusLocation:
    return CorpusLocation(document_index=d['document_index'],
                          character_index=d['character_index'])


@dataclass
class AudioTranscriptionChunk:
    """
    Minimum unit of transcription to be highlighted during an audiobook reading.
    The transcription is created by concatenating the chunks.
    """
    file_path: os.PathLike
    start_secs: float
    end_secs: float
    text: str

    def to_serializable_dict(self):
        return {
            'file_path': str(self.file_path),
            'start_secs': self.start_secs,
            'end_secs': self.end_secs,
            'text': self.text,
        }


def audiotranscriptionchunk_from_dict(json_as_dict: dict) -> AudioTranscriptionChunk:
    audio_file_path = Path(json_as_dict['file_path'])
    if not audio_file_path.is_file():
        raise ValueError("Invalid 'file_path' attribute; file does not exist: {}".format(audio_file_path))
    start_secs = float(json_as_dict['start_secs'])
    end_secs = float(json_as_dict['end_secs'])
    if start_secs < 0 or end_secs < 0 or start_secs > end_secs:
        raise ValueError("Invalid start_secs ({}) or end_secs ({})".format(start_secs, end_secs))
    text = json_as_dict['text'] or ""
    return AudioTranscriptionChunk(file_path=json_as_dict['file_path'],
                                   start_secs=start_secs, end_secs=end_secs,
                                   text=text)


@dataclass
class AudioTranscription:
    chunks: list[AudioTranscriptionChunk]

    def to_serializable_dict(self) -> dict:
        chunk_dicts = []
        for chunk in self.chunks:
            chunk_dicts.append(chunk.to_serializable_dict())
        return {
            'chunks': chunk_dicts
        }


def audiobooktranscription_from_dict(json_as_dict: dict) -> AudioTranscription:
    chunks = []
    for chunk_d in json_as_dict['chunks']:
        chunks.append(audiotranscriptionchunk_from_dict(chunk_d))
    return AudioTranscription(chunks=chunks)


def merge_audio_transcriptions(transcriptions: Iterable[AudioTranscription]):
    chunks_all = []
    for tr in transcriptions:
        chunks_all.extend(tr.chunks)
    return AudioTranscription(chunks=chunks_all)

