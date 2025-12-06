from .renderers_common import Renderer, RenderingFailedError
from ..common import AudioTranscription, CorpusLocation, TextSegment
from ..logging import LoggerFactory

from typing import Sequence
import os
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
from ebooklib import epub


class ReadaloudRenderer(Renderer):
    def __init__(self, loggerfactory: LoggerFactory, output_directory: os.PathLike,
                 original_epub_path: os.PathLike, document_index_to_href: dict[int, str],
                 original_epub_location_mappings: dict[str, list[int]]):
        """
        Args:
            document_index_to_href: mapping from document index to chapter's href
            original_epub_location_mappings: key is href, and list[int] maps plain text document's
                (document which corresponds to that href, given by `epub_href_to_document`)
                character index to that in the original epub document pointed by the href.
        """
        super().__init__()
        self.logger = loggerfactory.get_logger(self.__class__.__name__)
        if not output_directory:
            raise ValueError("output directory shall not be empty")
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.original_epub = epub.read_epub(original_epub_path)
        self.original_epub_name = Path(original_epub_path).stem
        self.document_index_to_href = document_index_to_href
        self.original_epub_location_mappings = original_epub_location_mappings

    def render(self,
               transcription: AudioTranscription,
               pages: Sequence[TextSegment],
               chunk_locations: Sequence[tuple[CorpusLocation, CorpusLocation]|None]):
        book = self._copy_ebook()
        audio_files = list(set(c.file_path for c in transcription.chunks))
        audio_file_names_in_epub = dict() # original audio file path -> audio file path in epub
        self.logger.info("Found %d audio file(s). Adding them to Readaloud book...", len(audio_files))
        for audio_index, audio_file_path in enumerate(tqdm(audio_files, desc="Adding audio files to Readaloud")):
            audio_ext = Path(audio_file_path).suffix
            if audio_ext.lower() not in ('.mp3', '.mpga', '.mp2a', '.mp2', '.m2a', '.m3a', '.m4a'):
                raise RenderingFailedError("Unsupported audio type: {}".format(audio_ext))
            file_name_in_epub = "{}{}".format(audio_index+1, audio_ext)
            path_in_epub = "Audio/" + file_name_in_epub
            media_type = "audio/mpeg"
            audio_file_names_in_epub[audio_file_path] = path_in_epub
            with open(audio_file_path, 'rb') as f:
                audio_item = epub.EpubItem(
                    uid=f"audio_{audio_index:02d}",
                    file_name=path_in_epub,
                    media_type=media_type,
                    content=f.read())
                book.add_item(audio_item)

        # Create SMIL
        def get_seq_html(_href: str, _items: list[str]) -> str:
            return (f"\t\t<seq epub:textref=\"../{_href}\">\n\t\t\t"
                    + "\n\t\t\t".join(_items)
                    + "\n\t\t</seq>")

        last_html_href = None
        seqs: list[str] = []
        cur_html_seq_items: list[str] = []
        chunk2fragid: dict[int, str] = dict()
        self.logger.info("Creating SMIL...")
        for chunk_idx, chunk in enumerate(transcription.chunks):
            chunk_st_ed = chunk_locations[chunk_idx]
            if chunk_st_ed is None:
                continue
            cur_html_href = self.document_index_to_href[chunk_st_ed[0].document_index]
            if cur_html_href != last_html_href and last_html_href is not None:
                seqs.append(get_seq_html(last_html_href, cur_html_seq_items))
                cur_html_seq_items.clear()

            fragment_id = "f{:05d}".format(chunk_idx+1)
            chunk2fragid[chunk_idx] = fragment_id
            cur_html_seq_items.append("<par>"
                + f"<text src=\"../{cur_html_href}#{fragment_id}\"/>"
                + f"<audio clipBegin=\"{self._secs_to_smil_timeformat(chunk.start_secs)}\""
                + f" clipEnd=\"{self._secs_to_smil_timeformat(chunk.end_secs)}\""
                + f" src=\"../{audio_file_names_in_epub[chunk.file_path]}\"/>"
                + "</par>")
            last_html_href = cur_html_href
        # Insert last seq
        if last_html_href is not None:
            seqs.append(get_seq_html(last_html_href, cur_html_seq_items))

        smil_content = ('<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n\t<body>\n'
            + "\n".join(seqs)
            + "\n\t</body>\n</smil>")
        smil_item = epub.EpubSMIL(file_name="MediaOverlays/sync.smil", content=smil_content)
        book.add_item(smil_item)

        # Insert timing info to chapters
        last_html_href = None
        cur_html = ""
        last_html_char_idx = 0 - 1
        cur_timed_html_chunks: list[str] = []
        timed_htmls: dict[str, str] = dict() # href -> timed html
        self.logger.info("Inserting timing info to EPUB HTML(s)")
        for chunk_idx, chunk in enumerate(transcription.chunks):
            chunk_st_ed = chunk_locations[chunk_idx]
            if chunk_st_ed is None:
                continue
            cur_html_href = self.document_index_to_href[chunk_st_ed[0].document_index]
            if cur_html_href != last_html_href:
                cur_html_item = self.original_epub.get_item_with_href(cur_html_href)
                if cur_html_item is None:
                    raise RenderingFailedError("Invalid href: {}".format(cur_html_href))
                last_html = cur_html
                cur_html = str(cur_html_item.content, encoding='utf-8')
                if last_html_href is not None: # add last html timed
                    if last_html_char_idx < len(last_html) - 1:
                        last_html_final_chunk = last_html[last_html_char_idx+1:]
                        cur_timed_html_chunks.append(last_html_final_chunk)
                    timed_htmls[last_html_href] = "".join(cur_timed_html_chunks)
                    cur_timed_html_chunks.clear()
                last_html_char_idx = 0 - 1

            # add current chunk and another preceding it
            cur_html_char_start_idx = self.original_epub_location_mappings[cur_html_href][chunk_st_ed[0].character_index]
            if chunk_st_ed[1].document_index != chunk_st_ed[0].document_index:
                self.logger.warning("Chunk %d/%d's document starts at %d but ends at %d",
                    chunk_idx+1, len(transcription.chunks), chunk_st_ed[0].document_index, chunk_st_ed[1].document_index)
                cur_html_char_end_idx = len(cur_html) - 1
            else:
                cur_html_char_end_idx = self.original_epub_location_mappings[cur_html_href][chunk_st_ed[1].character_index]
            if cur_html_char_start_idx > last_html_char_idx:
                preceding_html_chunk = cur_html[last_html_char_idx+1:cur_html_char_start_idx]
                cur_timed_html_chunks.append(preceding_html_chunk)
            if cur_html_char_start_idx >= last_html_char_idx:
                cur_html_chunk = cur_html[cur_html_char_start_idx:cur_html_char_end_idx+1]
                last_html_char_idx = cur_html_char_end_idx
                cur_chunk_fragid = chunk2fragid[chunk_idx]
                cur_timed_html_chunk = f"<span id={cur_chunk_fragid}>{cur_html_chunk}</span>"
                cur_timed_html_chunks.append(cur_timed_html_chunk)
            else:
                # Same text chunks may be repeated if a text chunk is split into multiple audio chunks,
                # for which case we could in the future implement merging audio chunks into a single fragment id.
                pass

            last_html_href = cur_html_href
        for href, timed_html in timed_htmls.items():
            book.get_item_with_href(href).set_content(timed_html.encode('utf-8')) # type: ignore
        self.logger.info("Finished adding timing infos to %d EPUB HTML(s)", len(timed_htmls))

        readaloud_fn = f"{self.original_epub_name}_readaloud.epub"
        readaloud_out_path = self.output_dir / readaloud_fn
        self.logger.info("Writing Readaloud EPUB to: %s", readaloud_out_path)
        epub.write_epub(readaloud_out_path, book)
        self.logger.info("Successfully written %s", readaloud_out_path)

    def _copy_ebook(self):
        """
        Copy contents of `self.original_epub` from which to derive the readaloud (EPUB3).
        """
        book = epub.EpubBook()
        book.set_title(self.original_epub.title)
        book.set_language(self.original_epub.language)
        book.metadata = {**self.original_epub.metadata}
        book.toc = list(self.original_epub.toc)
        book.spine = list(self.original_epub.spine)
        for original_epub_item in self.original_epub.get_items():
            if isinstance(original_epub_item, epub.EpubNcx):
                continue
            book.add_item(original_epub_item)
        return book

    def _secs_to_smil_timeformat(self, secs: float) -> str:
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.03f}"
