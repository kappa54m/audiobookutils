from ..common import AudioTranscription, CorpusLocation, TextSegment

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Mapping
import os
from pathlib import Path


@dataclass
class PageTiming:
    audio_file_path_at_page_start: Path
    audio_file_path_at_page_end: Path
    start_secs: float
    end_secs: float


class Renderer(ABC):
    @abstractmethod
    def render(self,
               transcription: AudioTranscription,
               pages: Sequence[TextSegment],
               chunk_locations: Sequence[tuple[CorpusLocation, CorpusLocation]|None]):
        """
        Args:
            chunk_locations: must have length equal to transcription.chunks.
                Each element corresponds to the start and end page locations of corresponding chunk.
        """
        pass

    def _calculate_pages_start_end_timings(self, transcription: AudioTranscription,
            chunk_locations: Sequence[tuple[CorpusLocation, CorpusLocation]|None]) -> Mapping[int, PageTiming]:
        """
        Returns:
            Mapping: page -> timing
        """
        chunks = transcription.chunks
        if len(chunks) == 0:
            raise ValueError("No chunks found")
        if len(chunks) != len(chunk_locations):
            raise ValueError("transcription.chunks and chunk_locations length mismatch")
        cur_page_start_secs = 0
        audio_file_path_at_page_start = Path(chunks[0].file_path)

        cur_page = 0
        page_timings = dict()
        for chunk_idx, chunk in enumerate(chunks):
            if chunk_locations[chunk_idx] is None:
                continue
            chunk_begin_loc, chunk_end_loc = chunk_locations[chunk_idx]
            if chunk_begin_loc.document_index != cur_page:
                if chunk_begin_loc.document_index < cur_page:
                    raise ValueError("Invalid begin loc of chunk {}/{}: {}".format(
                        chunk_idx+1, len(chunks), chunk_begin_loc.document_index))
                page_timings[cur_page] = PageTiming(audio_file_path_at_page_start=audio_file_path_at_page_start,
                                                    audio_file_path_at_page_end=Path(chunk.file_path),
                                                    start_secs=cur_page_start_secs,
                                                    end_secs=chunk.start_secs)
                audio_file_path_at_page_start = Path(chunk.file_path)
                cur_page_start_secs = chunk.start_secs
                cur_page = chunk_begin_loc.document_index
            elif chunk_idx == len(chunks) - 1:
                page_timings[cur_page] = PageTiming(audio_file_path_at_page_start=audio_file_path_at_page_start,
                                                    audio_file_path_at_page_end=Path(chunk.file_path),
                                                    start_secs=cur_page_start_secs,
                                                    end_secs=chunk.end_secs)
        return page_timings






