from .common import TextSegment, CorpusLocation
from .logging import LoggerFactory

from rapidfuzz.distance import Levenshtein
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import Sequence
from collections import namedtuple, defaultdict
import re
import string


class CorpusAligner(ABC):
    @abstractmethod
    def perform_alignment(self, source_corpus: Sequence[TextSegment], target_words: Sequence[str]) \
            -> dict[int, tuple[CorpusLocation, CorpusLocation]]:
        """
        Map each word in `target_words` to location pairs (start & end) in `source_corpus`
        Returns:
            index in `target_words` -> start and end index if correspondence exists
        """
        pass


Token = namedtuple('Token', ('norm_word', 'word', 'start_idx', 'end_idx'))


class GlobalNormalizedWordAligner(CorpusAligner):
    def __init__(self, loggerfactory: LoggerFactory, use_dp: bool=False, remove_spaces_around_matches: bool=True):
        """
        Args:
            use_dp: use the optimal dynamic programming algorithm for matching (may be slower)
            remove_spaces_around_matches: remove spaces around each word while matching
        """
        self.logger = loggerfactory.get_logger(self.__class__.__name__)
        self.use_dp = use_dp
        self.remove_spaces_around_matches = remove_spaces_around_matches

    """
    Word-based aligner based on rapidfuzz fuzzy matcher
    """
    def perform_alignment(self, source_corpus: Sequence[TextSegment], target_words: Sequence[str]) \
            -> dict[int, tuple[CorpusLocation, CorpusLocation]]:
        """
        Align each target word to start/end locations in source corpus
        Args:
            target_words: empty words are not allowed
        Returns:
            dict {target word index: (source start loc, source end loc)}
        """
        for i, w in enumerate(target_words):
            if not w: raise ValueError("Empty target word at {}/{}: {}".format(i+1, len(target_words), w))

        source_corpus_merged, source_corpus_mergedidx2corpusloc, _ = self._merge_corpus(source_corpus)
        target_words_merged, target_words_mergedidx2corpusloc, target_words_intervals  = self._merge_corpus(target_words, delimiter=" ")

        # Get source <-> target match intervals
        alignment = dict()
        opcodes = Levenshtein.opcodes(source_corpus_merged, target_words_merged)

        self.logger.info("Parsing matches...")
        match_stats = defaultdict(lambda: 0)
        source_match_intervals, target_match_intervals = [], []
        spaces_pat = re.compile(r"^(\s*)([^\s].*[^\s])(\s*)$")

        def remove_spaces(_s: str, _og_interval: tuple[int, int]) -> tuple[int, int]:
            """
            Remove leading and training spaces of substring of `_s` in `_og_interval` ((start, end); end is exclusive)
            Returns: new interval without leading & trailing spaces (start, end), where end is exclusive
            """
            m = spaces_pat.search(_s[_og_interval[0]:_og_interval[1]])
            if m is not None:
                rel = m.span(2)
                return (_og_interval[0] + rel[0], _og_interval[0] + rel[1])
            else:
                # Must be all spaces, return empty interval
                return (_og_interval[0], _og_interval[0])

        for tag, source_idx_begin, source_idx_end, target_idx_begin, target_idx_end in opcodes:
            match tag:
                case 'equal' | 'replace':
                    assert isinstance(source_idx_begin, int) and isinstance(source_idx_end, int)
                    assert isinstance(target_idx_begin, int) and isinstance(target_idx_end, int)
                    assert source_idx_end - source_idx_begin > 0 and target_idx_end - target_idx_begin > 0
                    if self.remove_spaces_around_matches:
                        source_idx_begin, source_idx_end = remove_spaces(source_corpus_merged, (source_idx_begin, source_idx_end))
                        target_idx_begin, target_idx_end = remove_spaces(target_words_merged, (target_idx_begin, target_idx_end))
                    if source_idx_end - source_idx_begin > 0 and target_idx_end - target_idx_begin > 0:
                        source_match_intervals.append((source_idx_begin, source_idx_end - 1))
                        target_match_intervals.append((target_idx_begin, target_idx_end - 1))
                case 'insert':
                    pass
                case 'delete':
                    pass
                case _:
                    raise ValueError("Unknown alignment tag: '{}'".format(tag))
            match_stats[tag] += 1
        self.logger.info("Word match stats (#target words: %d): #matches: %d, %s",
                         len(target_words), len(source_match_intervals), {**match_stats})

        # Map target words to target (str), then to source
        if self.use_dp:
            # Must be done in batches due to memory usage
            aligned_target_match_intervals_all = dict() # target word idx -> target match idx
            dp_batch_size = 10000
            target_match_int_curbatch_idx = 0
            cur_target_word_idx = 0
            for cur_target_word_idx in tqdm(range(0, len(target_words), dp_batch_size), desc="Target word<->match alignment (DP) iteration"):
                next_target_word_idx = min(len(target_words) - 1, cur_target_word_idx + dp_batch_size)
                cur_target_words_end_idx = target_words_intervals[next_target_word_idx][1]
                target_match_int_nextbatch_idx = -1
                for target_match_int_idx, target_match_int in enumerate(target_match_intervals):
                    if target_match_int[1] > cur_target_words_end_idx or target_match_int_idx == len(target_match_intervals) - 1:
                        target_match_int_nextbatch_idx = max(0, target_match_int_idx - 1)
                        break
                target_match_intervals_curbatch = target_match_intervals[target_match_int_curbatch_idx:target_match_int_nextbatch_idx+1+1]
                target_word_intervals_curbatch = target_words_intervals[cur_target_word_idx:next_target_word_idx+1]

                aligned_target_match_intervals_curbatch, align_score_curbatch = \
                    self._align_intervals_dp(target_word_intervals_curbatch, target_match_intervals_curbatch)
                target_word_intervals_sum_curbatch = sum(max(0, i[1] - i[0]) for i in target_word_intervals_curbatch)
                self.logger.info("Batch %d-%d alignment score: %f/%f (%.02f%%)",
                    cur_target_word_idx, next_target_word_idx, align_score_curbatch, target_word_intervals_sum_curbatch,
                    100*align_score_curbatch/max(1, target_word_intervals_sum_curbatch))
                for rel_target_word_idx, rel_target_match_idx in enumerate(aligned_target_match_intervals_curbatch):
                    aligned_target_match_intervals_all[cur_target_word_idx + rel_target_word_idx] = \
                        target_match_int_curbatch_idx + rel_target_match_idx

                target_match_int_curbatch_idx = target_match_int_nextbatch_idx 
                cur_target_word_idx += dp_batch_size
        else:
            aligned_target_match_intervals_all, overlaps = self._align_intervals_twopointers(target_words_intervals, target_match_intervals)
            target_word_intervals_sum = sum(max(0, i[1] - i[0]) for i in target_words_intervals)
            self.logger.info("Alignment score: %f/%f (%.02f%%)",
                overlaps, target_word_intervals_sum, 100*overlaps/max(1, target_word_intervals_sum))

        for target_word_i in range(len(target_words)):
            target_match_int_idx = aligned_target_match_intervals_all[target_word_i]
            source_match_int = source_match_intervals[target_match_int_idx]
            alignment[target_word_i] = (source_corpus_mergedidx2corpusloc[source_match_int[0]],
                                        source_corpus_mergedidx2corpusloc[source_match_int[1]])
        
        return alignment
    
    def _merge_corpus(self, corpus: Sequence[TextSegment]|Sequence[str], delimiter="\n") -> tuple[str, dict[int, CorpusLocation], Sequence[tuple]]:
        """
        Merge corpus into one document, and return a map: index in merged doc -> CorpusLocation
        Args:
            delimiter: Delimiter between documents in corpus
        Returns:
            merged_corpus: Corpus merged, as a single string
            mergedidx2corpusloc: dict: index in merged document -> corpus location
            merged_intervals: List of (start_idx, end_idx) where start_idx and end_idx
                are starting and ending (inclusive) indices of a document.
        """
        if len(corpus) == 0:
            raise ValueError("Empty corpus")
        texts_all: list[str] = []
        for doc in corpus:
            if isinstance(doc, TextSegment):
                txt = doc.text
            else:
                txt = doc
            texts_all.append(txt)

        mergedidx2corpusloc = dict()
        merged_corpus = ""
        merged_intervals = []
        for doc_idx, doc in enumerate(texts_all):
            last_char_in_doc_idx = -1
            for char_idx in range(len(doc)):
                merged_idx = len(merged_corpus) + char_idx
                assert merged_idx not in mergedidx2corpusloc
                mergedidx2corpusloc[merged_idx] = CorpusLocation(document_index=doc_idx, character_index=char_idx)
                last_char_in_doc_idx = char_idx
            merged_intervals.append((len(merged_corpus), len(merged_corpus) + len(doc) - 1))
            merged_corpus += doc
            if doc_idx < len(texts_all):
                merged_corpus_last_idx = len(merged_corpus) - 1
                merged_corpus += delimiter
                assert last_char_in_doc_idx >= 0
                assert len(merged_corpus) not in mergedidx2corpusloc
                for i in range(0, len(delimiter)):
                    mergedidx2corpusloc[merged_corpus_last_idx + 1 + i] = \
                        CorpusLocation(document_index=doc_idx, character_index=last_char_in_doc_idx)
        assert len(merged_corpus) == len(mergedidx2corpusloc)
        return merged_corpus, mergedidx2corpusloc, merged_intervals

    def _align_intervals_dp(self, intervals_a: Sequence[Sequence], intervals_b: Sequence[Sequence]) -> tuple[list[int], int]:
        """
        Finds the best overlaps for each interval in A from B
        Time and memory complexity are both O(|A|x|B|) so use at own caution.
        Args:
            intervals_a: list of 2-tuples (A)
            intervals_b: list of 2-tuples (B)
        Returns:
            - list of B indices for each matched interval in A. Has same length as A and is monotonically increasing.
            - Overlap count
        """
        na = len(intervals_a)
        nb = len(intervals_b)
        if na == 0 or nb == 0:
            return [], 0

        self.logger.debug("Allocating array of shape %dx%d for intervals alignment (DP)", na+1, nb+1)
        dp = np.zeros((na+1, nb+1), dtype=np.float32)
        
        for i in tqdm(range(na - 1, -1, -1), desc="align intervals - DP"):
            for j in range(nb - 1, -1, -1):
                overlap = self._calculate_overlap(intervals_a[i], intervals_b[j])
                opt1_score = overlap + dp[i+1, j]
                opt2_score = dp[i, j+1]
                dp[i][j] = max(opt1_score, opt2_score)

        results = []
        i, j = 0, 0
        while i < na and j < nb:
            overlap = self._calculate_overlap(intervals_a[i], intervals_b[j])
            opt1_score = overlap + dp[i+1, j]
            opt2_score = dp[i, j+1]
            if opt1_score >= opt2_score:
                results.append(j)
                i += 1
            else:
                j += 1
        return results, dp[0, 0]

    def _align_intervals_twopointers(self, intervals_a: Sequence[Sequence], intervals_b: Sequence[Sequence]) -> tuple[list[int], int]:
        """
        Finds the overlaps for each interval in A from B
        This is a less accurate, greedy version of `_align_intervals_dp` that should run in O(|A| + |B|) time in most cases.
        """
        if len(intervals_a) == 0 or len(intervals_b) == 0:
            raise ValueError("Intervals shall not be empty")

        na, nb = len(intervals_a), len(intervals_b)
        results = na * [-1]
        last_j = 0
        overlaps = 0
        for i in tqdm(range(na), desc="align intervals - twopointers"):
            for j in range(last_j, nb):
                overlap = self._calculate_overlap(intervals_a[i], intervals_b[j])
                if overlap > 0:
                    last_j = j
                    results[i] = last_j
                    overlaps += overlap
                    break

        if all(r == -1 for r in results):
            self.logger.warning("Failed to match any interval in A (#=%d) to B (#=%d)", na, nb)
            results = na * [0]
        else:
            # Fill unset (-1) values
            for i in range(na):
                if results[i] < 0:
                    results[i] = 0
                else:
                    break

            last_nonneg_val = results[0]
            assert last_nonneg_val >= 0
            for i in range(na):
                if results[i] < 0:
                    results[i] = last_nonneg_val
                else:
                    last_nonneg_val = results[i]
        return results, overlaps

    def _tokenize_document(self, text: str) -> list[Token]:
        tokens = []
        for m in re.finditer(r'\S+', text):
            word = m.group(0)
            norm_word = word.lower().strip(string.punctuation)
            if not norm_word:
                continue
            tokens.append(Token(norm_word=norm_word, word=word, start_idx=m.start(), end_idx=m.end()))
        return tokens

    def _calculate_overlap(self, _int_a: Sequence[int], _int_b: Sequence[int]) -> int:
        st = max(_int_a[0], _int_b[0])
        ed = min(_int_a[1], _int_b[1])
        return max(0, ed - st)
