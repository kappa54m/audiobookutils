from .common import AudioTranscription, TextSegment, Ebook, CorpusLocation, ebook_from_dict, corpuslocation_from_dict, merge_audio_transcriptions
from .logging import LoggerFactory
from .sequence_alignment import CorpusAligner
from .pager import PageOptions, Pager
from .renderers import Renderer, SRTRenderer

from ebooklib import epub
from bs4 import BeautifulSoup
import hydra
from omegaconf import DictConfig

from typing import Sequence, Iterable
import os
from pathlib import Path
from dataclasses import dataclass
import sys
import json


@dataclass
class EbookTranscriptionMatch:
    audio_file: os.PathLike
    start_secs: float
    end_secs: float
    start_location: CorpusLocation
    end_location: CorpusLocation

    def to_serializable_dict(self):
        return {
            'audio_file': str(self.audio_file),
            'start_secs': float(self.start_secs),
            'end_secs': float(self.end_secs),
            'start_location': self.start_location.to_serializable_dict(),
            'end_location': self.end_location.to_serializable_dict()
        }


def ebook_transcription_match_from_dict(d: dict) -> EbookTranscriptionMatch:
    return EbookTranscriptionMatch(
        audio_file=Path(d['audio_file']),
        start_secs=float(d['start_secs']),
        end_secs=float(d['end_secs']),
        start_location=corpuslocation_from_dict(d['start_location']),
        end_location=corpuslocation_from_dict(d['end_location'])
    )


@dataclass
class EbookTranscriptionMatchResult:
    book: Ebook
    matches: Sequence[EbookTranscriptionMatch]

    def to_serializable_dict(self):
        return {
            'book': self.book.to_serializable_dict(),
            'matches': [m.to_serializable_dict() for m in self.matches]
        }


def ebook_transcription_match_result_from_dict(d: dict) -> EbookTranscriptionMatchResult:
    ebook = ebook_from_dict(d['book'])
    matches = []
    for m in d['matches']:
        matches.append(ebook_transcription_match_from_dict(m))
    return EbookTranscriptionMatchResult(book=ebook, matches=matches)


class Engine:
    def __init__(self, loggerfactory: LoggerFactory, aligner: CorpusAligner, renderer: Renderer):
        self.logger = loggerfactory.get_logger(self.__class__.__name__)
        self.aligner = aligner
        self.renderer = renderer

    def match_ebook_and_audio_transcriptions(self, ebook_path: os.PathLike, transcriptions: Sequence[AudioTranscription]) -> EbookTranscriptionMatchResult:
        book = self._load_ebook(ebook_path)
        all_chapters = list(book.chapters)
        matches = []
        for transcription_idx, transcription in enumerate(transcriptions):
            transcription_words = [chunk.text for chunk in transcription.chunks]
            self.logger.info("Performing alignment of transcription %d/%d: %s (#words: %d)",
                             transcription_idx+1, len(transcriptions), ebook_path, len(transcription_words))
            alignment = self.aligner.perform_alignment(source_corpus=all_chapters, target_words=transcription_words)
            for word_idx, (start_loc, end_loc) in alignment.items():
                chunk = transcription.chunks[word_idx]
                matches.append(EbookTranscriptionMatch(
                    audio_file=chunk.file_path,
                    start_secs=chunk.start_secs, end_secs=chunk.end_secs,
                    start_location=start_loc, end_location=end_loc))
        return EbookTranscriptionMatchResult(book=book, matches=matches)

    def render(self, book: Ebook, matches: Sequence[EbookTranscriptionMatch], transcriptions: Iterable[AudioTranscription], page_options: PageOptions):
        pager = Pager()
        self.logger.info("Converting Ebook (%s) chapters (#=%d) to pages...", book.title, len(book.chapters))
        paging_result = pager.to_pages(chapters=book.chapters, options=page_options)

        transcription = merge_audio_transcriptions(transcriptions)
        chunk_locations_in_pages = []
        n_unmatched_chunks = 0
        for i_chunk, match in enumerate(matches):
            chunk_st_chap_loc = match.start_location
            chunk_ed_chap_loc = match.end_location
            if chunk_st_chap_loc in paging_result.chaps2pages and \
                chunk_ed_chap_loc in paging_result.chaps2pages:
                chunk_st_page_loc = paging_result.chaps2pages[chunk_st_chap_loc]
                chunk_ed_page_loc = paging_result.chaps2pages[chunk_ed_chap_loc]
                chunk_locations_in_pages.append((chunk_st_page_loc, chunk_ed_page_loc))
            else:
                n_unmatched_chunks += 1
                chunk_locations_in_pages.append(None)
        self.logger.info("Start rendering using %s...", self.renderer)
        self.renderer.render(transcription=transcription, pages=paging_result.pages,
                             chunk_locations=chunk_locations_in_pages)
        self.logger.info("Rendering finished")

    def _load_ebook(self, ebook_path: os.PathLike) -> Ebook:
        """
        Returns:
            Tuple of title and list of chapters
        """
        if not Path(ebook_path).is_file():
            raise ValueError("Ebook path does not point to a file: {}".format(ebook_path))
        if Path(ebook_path).suffix.lower() != '.epub':
            raise ValueError("Unable to load ebook '{}'. Expected EPUB format.".format(ebook_path))
        self.logger.info("Loading ebook: %s", ebook_path)
        book = epub.read_epub(ebook_path)
        title = book.title
        author = "Unknown"
        for namespace, metadata in book.metadata.items():
            if 'creator' in metadata:
                if isinstance(metadata['creator'], str):
                    author = metadata['creator']
                else:
                    cr = metadata['creator'][0]
                    if isinstance(cr, str):
                        author = cr
                    else:
                        author = str(cr[0])
                break
        chaps = self._extract_chapters(book)
        self.logger.info("Extracted %d chapter(s) from ebook (title: %s, author: %s, file: %s)",
                         len(chaps), title, author, ebook_path)
        return Ebook(title=book.title, author=author, chapters=chaps)

    def _extract_chapters(self, book: epub.EpubBook) -> list[TextSegment]:
        """
        Read table of contents and return list of chapters
        """
        chap_links = []
        for item in book.toc:
            if isinstance(item, epub.Link):
                self.logger.debug(f"Chapter Link: Title='{item.title}', Href='{item.href}'")
                chap_links.append(item)
            elif isinstance(item, epub.Section):
                self.logger.debug(f"Section: Title='{item.title}'")
                for sub_item in item.children: # type: ignore
                    if isinstance(sub_item, epub.Link):
                        self.logger.debug(f"  Sub-Chapter Link: Title='{sub_item.title}', Href='{sub_item.href}'")
                        chap_links.append(sub_item)

        chaps = []
        for ichap in range(len(chap_links)):
            chap_epubhtml = book.get_item_with_href(chap_links[ichap].href)
            if chap_epubhtml is None:
                raise ValueError("Failed to parse chapter {}/{}: {} ({})".format(
                    ichap+1, len(chap_links), chap_links[ichap].title, chap_links[ichap].href))
            chaps.append(self._parse_chapter_epubhtml(title=chap_links[ichap].title, epubhtml=chap_epubhtml))
        return chaps

    def _parse_chapter_epubhtml(self, title: str, epubhtml: epub.EpubHtml) -> TextSegment:
        chap_soup = BeautifulSoup(epubhtml.content, 'html.parser')
        ch_title_e = chap_soup.select_one('title')
        if ch_title_e is not None:
            ch_title_e.decompose()
        ch_title_e2 = chap_soup.select_one('h1.title')
        if ch_title_e2 is not None:
            ch_title_e2.decompose()
        ch_text = chap_soup.text
        return TextSegment(name=title, text=ch_text)


@hydra.main(version_base=None, config_path='../conf', config_name='match_book_and_audio')
def main(conf: DictConfig):
    logging_conf = conf['logging']

    lf = LoggerFactory(console_logging_level=logging_conf['console_logging_level'],
                       do_file_logging=logging_conf['do_file_logging'],
                       file_logging_level=logging_conf['file_logging_level'],
                       file_logging_dir=logging_conf['file_logging_dir'])
    logger = lf.get_logger(__name__)

    output_dir = conf['output_dir']
    if not output_dir:
        print("No output dir specified. Aborting.", file=sys.stderr)
        exit(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Output directory: %s", output_dir)

    from .sequence_alignment import GlobalNormalizedWordAligner
    from .common import audiobooktranscription_from_dict


    aligner = GlobalNormalizedWordAligner(loggerfactory=lf, use_dp=conf['aligner_use_dp'])
    renderer = SRTRenderer(loggerfactory=lf, output_directory=output_dir)
    engine = Engine(loggerfactory=lf, aligner=aligner, renderer=renderer)

    ebook_path = conf['ebook_path']
    if not os.path.isfile(ebook_path):
        print("Invalid ebook path; file does not exist: %s" % ebook_path, file=sys.stderr)
        exit(1)
    logger.info("Ebook path: %s", ebook_path)

    paging_conf = conf['paging']

    # Load audio transcription(s)
    transcription_result_paths = conf['transcription_result_paths']
    transcriptions = []
    logger.info("Transcription path(s) (#=%d): %s", len(transcription_result_paths), transcription_result_paths)
    for i, transcription_path in enumerate(transcription_result_paths):
        if not transcription_path or not os.path.isfile(transcription_path):
            print("Invalid transcription file (%d/%d): '%s'. Aborting"
                % (i+1, len(transcription_result_paths), transcription_path), file=sys.stderr)
            exit(1)

        with open(transcription_path, encoding='utf-8') as f:
            transcription_dict = json.load(f)
            transcriptions.append(audiobooktranscription_from_dict(transcription_dict))

    # Match ebook and audio transcription(s)
    match_result_path = output_dir / "ebook_and_audio_transcriptions_match.json"
    if not match_result_path.is_file():
        logger.info("Matching ebook '%s' with %d audio transcription(s)", ebook_path, len(transcriptions))
        match_result = engine.match_ebook_and_audio_transcriptions(ebook_path=ebook_path, transcriptions=transcriptions)
        logger.info("Writing match results to: %s", match_result_path)
        with open(match_result_path, 'w', encoding='utf-8') as f:
            json.dump(match_result.to_serializable_dict(), f, indent=2, ensure_ascii=False)
    else:
        logger.info("Ebook '%s' with %d audio transcriptions(s) match data already exists (%s). Loading...",
                    ebook_path, len(transcriptions), match_result_path)
        with open(match_result_path, encoding='utf-8') as f:
            match_result_dict = json.load(f)
        match_result = ebook_transcription_match_result_from_dict(match_result_dict)

    # Render
    page_options = PageOptions(max_lines=paging_conf['max_lines'],
                               max_chars_per_line=paging_conf['max_characters_per_line'])
    logger.info("Start rendering book (%s) with options: %s", match_result.book.title, page_options)
    engine.render(book=match_result.book, matches=match_result.matches,
                  transcriptions=transcriptions, page_options=page_options)
    logger.info("Finished")


if __name__ == '__main__':
    main()
