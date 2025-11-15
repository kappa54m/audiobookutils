from .renderers_common import Renderer
from ..common import AudioTranscription, CorpusLocation, TextSegment
from ..logging import LoggerFactory

from typing import Sequence
import os
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class SubtitleBlock:
    start_secs: float
    end_secs: float
    text: str


class SRTRenderer(Renderer):
    def __init__(self, loggerfactory: LoggerFactory, output_directory: os.PathLike):
        super().__init__()
        self.logger = loggerfactory.get_logger(self.__class__.__name__)
        if not output_directory:
            raise ValueError("output directory shall not be empty")
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def render(self,
               transcription: AudioTranscription,
               pages: Sequence[TextSegment],
               chunk_locations: Sequence[tuple[CorpusLocation, CorpusLocation]|None]):
        pages_st_ed_timings = self._calculate_pages_start_end_timings(
            transcription=transcription, chunk_locations=chunk_locations)
        chunks = transcription.chunks
        cur_chunk_idx = 0
        cur_chunk = chunks[cur_chunk_idx]
        cur_chunk_fp = self._canon_path(cur_chunk.file_path)
        sub_blocks = []
        file_index = 0
        last_page_audio_fp = None
        for page, page_content in enumerate(tqdm(pages, desc="Processing pages for SRT")):
            if page not in pages_st_ed_timings:
                continue
            if page == 0:
                sub_blocks.append(self._get_subtitle_block(start_secs=0, end_secs=cur_chunk.start_secs,
                                                           page=page_content.text, highlight_range=None))
            page_timings = pages_st_ed_timings[page]
            page_audio_fp = self._canon_path(page_timings.audio_file_path_at_page_start)
            while cur_chunk.end_secs <= page_timings.end_secs \
                and cur_chunk_fp == page_audio_fp:
                highlight_range = None
                if chunk_locations[cur_chunk_idx] is not None:
                    highlight_range = (chunk_locations[cur_chunk_idx][0].character_index, # type: ignore
                                       chunk_locations[cur_chunk_idx][1].character_index) # type: ignore
                end_secs = chunks[cur_chunk_idx+1].start_secs if cur_chunk_idx < len(chunks) - 1 else cur_chunk.end_secs
                sub_blocks.append(self._get_subtitle_block(
                    start_secs=cur_chunk.start_secs, end_secs=end_secs, page=page_content.text,
                    highlight_range=highlight_range))
                if cur_chunk_idx >= len(chunks) - 1:
                    break
                else:
                    cur_chunk_idx += 1
                    cur_chunk = chunks[cur_chunk_idx]
                    cur_chunk_fp = self._canon_path(cur_chunk.file_path)

            if cur_chunk_fp != page_audio_fp or page == len(pages) - 1:
                fn = Path(last_page_audio_fp or page_audio_fp).stem + ".srt"
                save_path = self.output_dir / fn
                self.logger.info("Saving subtitle file %d to: %s", file_index+1, save_path)
                self._save_subtitles(sub_blocks=sub_blocks, save_path=save_path)
                sub_blocks.clear()
                file_index += 1
                
                while cur_chunk_idx < len(chunks) - 1:
                    cur_chunk_idx += 1
                    cur_chunk = chunks[cur_chunk_idx]
                    cur_chunk_fp = self._canon_path(cur_chunk.file_path)
                    if cur_chunk_fp == page_audio_fp:
                        break
            last_page_audio_fp = page_audio_fp

    def _save_subtitles(self, sub_blocks: Sequence[SubtitleBlock], save_path: os.PathLike):
        save_path = Path(save_path)
        if save_path.suffix.lower() != ".srt":
            raise ValueError("Unsupported subtitle format: {}".format(save_path))
        save_path.parent.mkdir(exist_ok=True, parents=True)
        lines = []
        for i_block, block in enumerate(sub_blocks):
            lines.append(str(i_block + 1))
            st = self._secs_to_srt_timeformat(block.start_secs)
            ed = self._secs_to_srt_timeformat(block.end_secs)
            lines.append(f"{st} --> {ed}")
            lines.append(block.text)
            lines.append("")
        
        self.logger.info("Writing %d line(s) to: %s", len(lines), save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    def _secs_to_srt_timeformat(self, secs: float) -> str:
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:" + f"{s:02.03f}".replace(".", ",")

    def _get_subtitle_block(self, start_secs: float, end_secs: float,
        page: str, highlight_range: tuple[int, int]|None) -> SubtitleBlock:
        """
        Args:
            highlight_range: Optional range to highlight (start and end index, both inclusive)
        """
        assert start_secs <= end_secs
        if not highlight_range:
            text = page
        else:
            #text1 = page[:highlight_range[0]].rstrip()
            #text_to_highlight = page[highlight_range[0]:highlight_range[1]+1].strip()
            #text2 = page[highlight_range[1]+1:].lstrip()
            #text = f"{text1} <u>{text_to_highlight}</u> {text2}"
            text1 = page[:highlight_range[0]]
            text_to_highlight = page[highlight_range[0]:highlight_range[1]+1]
            text2 = page[highlight_range[1]+1:]

            text = text1
            highlight_startswithspace = text_to_highlight.startswith(" ")
            highlight_endswithspace = text_to_highlight.endswith(" ")
            if highlight_startswithspace:
                text += " "
                text_to_highlight = text_to_highlight[1:]
            if highlight_endswithspace:
                text_to_highlight = text_to_highlight[:-1]
            text += "<u>" + text_to_highlight + "</u>"
            if highlight_endswithspace:
                text += " "
            text += text2
        return SubtitleBlock(text=text, start_secs=start_secs, end_secs=end_secs)

    def _canon_path(self, path: os.PathLike) -> str:
        return os.path.realpath(path)

