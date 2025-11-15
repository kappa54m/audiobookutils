from .common import TextSegment, CorpusLocation

from typing import Sequence, Mapping
import re
from dataclasses import dataclass


@dataclass
class PageOptions:
    """
    Defines the layout constraints for a page.
    """
    max_lines: int = 40
    max_chars_per_line: int = 80


@dataclass
class PagingResult:
    pages: Sequence[TextSegment]
    pages2chaps: Mapping[CorpusLocation, CorpusLocation]
    chaps2pages: Mapping[CorpusLocation, CorpusLocation]


class Pager:
    """
    Implements pagination for a corpus of TextSegments (chapters).
    
    Generates pages and the corresponding character-level mappings
    between the source chapters and the generated pages.
    """

    def to_pages(self, chapters: Sequence[TextSegment], options: PageOptions) -> PagingResult:
        """
        Converts a sequence of chapters into a sequence of pages.

        Args:
            chapters: The list of TextSegment objects, each representing a chapter.
            options: The PageOptions defining layout constraints.

        Returns:
            - "pages" (Sequence[TextSegment]): The list of generated pages.
            - "pages2chaps" (Mapping[...]): The forward map (page_loc -> chapter_loc).
            - "chaps2pages" (Mapping[...]): The inverse map (chapter_loc -> page_loc).
        """
        
        # --- Initialization ---
        pages: list[TextSegment] = []
        page_to_chapter_map: dict[CorpusLocation, CorpusLocation] = {}
        chapter_to_page_map: dict[CorpusLocation, CorpusLocation] = {}

        global_page_index = 0
        
        # This list holds the data for the page currently being built.
        # Each item is (line_string, {char_index_in_line -> CorpusLocation_in_chapter})
        current_page_data: list[tuple[str, dict[int, CorpusLocation]]] = []

        # --- Chapter Iteration ---
        for chapter_index, chapter in enumerate(chapters):
            # Location of the first character of the chapter's text
            # We map the title to this location.
            first_char_loc = CorpusLocation(chapter_index, 0)
            
            # --- 1. Handle Chapter Title ---
            if chapter.name:
                # Add title line
                if len(current_page_data) >= options.max_lines:
                    # Page is full before adding title, finalize it
                    self._finalize_page(
                        pages, current_page_data, global_page_index,
                        page_to_chapter_map, chapter_to_page_map
                    )
                    global_page_index += 1
                    current_page_data = []
                
                # Add title, with mapping for each char to first char of text
                title_mappings: dict[int, CorpusLocation] = {}
                for i in range(len(chapter.name)):
                    title_mappings[i] = first_char_loc
                current_page_data.append((chapter.name, title_mappings))

                # Add blank line
                if len(current_page_data) >= options.max_lines:
                    # Page is full after title, finalize it
                    self._finalize_page(
                        pages, current_page_data, global_page_index,
                        page_to_chapter_map, chapter_to_page_map
                    )
                    global_page_index += 1
                    current_page_data = []
                # Add blank line (no mapping)
                current_page_data.append(("", {}))

            # --- 2. Process Chapter Text ---
            text = chapter.text
            
            # Split text into words AND whitespace. This is key for tracking
            # original character indices, including spaces and newlines.
            # e.g., "Hello\nworld" -> ["Hello", "\n", "world"]
            tokens = re.split(r'(\s+)', text)
            
            current_char_in_chapter = 0
            
            current_line_text = ""
            current_line_mappings: dict[int, CorpusLocation] = {}

            for token in tokens:
                if not token:
                    continue
                
                token_start_in_chapter = current_char_in_chapter
                token_len = len(token)
                
                # Check token type
                is_newline = '\n' in token
                is_space = token.isspace()

                if is_newline:
                    # Hard newline in source. Finish the current line.
                    if len(current_page_data) >= options.max_lines:
                        self._finalize_page(
                            pages, current_page_data, global_page_index,
                            page_to_chapter_map, chapter_to_page_map
                        )
                        global_page_index += 1
                        current_page_data = []
                    
                    # Add the line before the newline
                    current_page_data.append((current_line_text, current_line_mappings))
                    
                    # Store the mapping for the newline itself.
                    # We map it to the newline char in the source.
                    # We represent this as a new, empty line that maps its
                    # (implied) newline char back to the source newline.
                    # This is complex. A simpler way: map the output
                    # newline (from `join`) to the source newline.
                    # We'll do this in _finalize_page by passing the source
                    # location of the character that caused the break.
                    
                    # For now, just start a new line.
                    current_line_text = ""
                    current_line_mappings = {}

                elif is_space:
                    # Whitespace (but not a newline). Try to add it.
                    if len(current_line_text) + token_len <= options.max_chars_per_line:
                        # It fits. Add it and its mappings.
                        for i in range(token_len):
                            line_idx = len(current_line_text) + i
                            chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + i)
                            current_line_mappings[line_idx] = chap_loc
                        current_line_text += token
                    else:
                        # Space doesn't fit. Wrap.
                        if len(current_page_data) >= options.max_lines:
                            self._finalize_page(
                                pages, current_page_data, global_page_index,
                                page_to_chapter_map, chapter_to_page_map
                            )
                            global_page_index += 1
                            current_page_data = []
                        
                        current_page_data.append((current_line_text, current_line_mappings))
                        current_line_text = "" # Eat the space, start new line
                        current_line_mappings = {}

                else:
                    # It's a word.
                    if not current_line_text:
                        # Word starts a new line.
                        if token_len <= options.max_chars_per_line:
                            # Word fits on a line by itself.
                            current_line_text = token
                            current_line_mappings = {}
                            for i in range(token_len):
                                chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + i)
                                current_line_mappings[i] = chap_loc
                        else:
                            # Word is longer than a line. Must break it.
                            start = 0
                            while start < token_len:
                                if len(current_page_data) >= options.max_lines:
                                    self._finalize_page(
                                        pages, current_page_data, global_page_index,
                                        page_to_chapter_map, chapter_to_page_map
                                    )
                                    global_page_index += 1
                                    current_page_data = []

                                chars_to_take = options.max_chars_per_line
                                is_last_part = (start + chars_to_take) >= token_len
                                
                                part_text = ""
                                part_mappings = {}
                                part_len = 0
                                
                                if not is_last_part:
                                    # Not the last part, so hyphenate
                                    chars_to_take = options.max_chars_per_line - 1 # Room for '-'
                                    part_text = token[start : start + chars_to_take] + "-"
                                    part_len = len(part_text)
                                    
                                    # Map the word characters
                                    for i in range(part_len - 1):
                                        chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + i)
                                        part_mappings[i] = chap_loc
                                    
                                    # Map the hyphen '-' to the last char it represents (e.g., 'r' in 'per-')
                                    hyphen_map_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + chars_to_take - 1)
                                    part_mappings[part_len - 1] = hyphen_map_loc
                                else:
                                    # Last part of the word. No hyphen.
                                    part_text = token[start:]
                                    part_len = len(part_text)
                                    for i in range(part_len):
                                        chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + i)
                                        part_mappings[i] = chap_loc
                                
                                current_page_data.append((part_text, part_mappings))
                                start += chars_to_take
                            
                            # Word is fully processed, ensure current line is empty
                            current_line_text = ""
                            current_line_mappings = {}
                            
                    elif len(current_line_text) + token_len <= options.max_chars_per_line:
                        # Word fits on the current line (which has content)
                        # The previous token must have been a space,
                        # which is already in current_line_text.
                        current_line_text += token
                        for i in range(token_len):
                            line_idx = len(current_line_text) - token_len + i
                            chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + i)
                            current_line_mappings[line_idx] = chap_loc
                    
                    else:
                        # Word doesn't fit on current line. Wrap.
                        if len(current_page_data) >= options.max_lines:
                            self._finalize_page(
                                pages, current_page_data, global_page_index,
                                page_to_chapter_map, chapter_to_page_map
                            )
                            global_page_index += 1
                            current_page_data = []
                        
                        # Add the previous line
                        current_page_data.append((current_line_text, current_line_mappings))
                        
                        # Start new line with this word (repeat logic from 'if not current_line_text')
                        if token_len <= options.max_chars_per_line:
                            current_line_text = token
                            current_line_mappings = {}
                            for i in range(token_len):
                                chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + i)
                                current_line_mappings[i] = chap_loc
                        else:
                            # Word is longer than a line. Break it.
                            start = 0
                            while start < token_len:
                                if len(current_page_data) >= options.max_lines:
                                    self._finalize_page(
                                        pages, current_page_data, global_page_index,
                                        page_to_chapter_map, chapter_to_page_map
                                    )
                                    global_page_index += 1
                                    current_page_data = []

                                chars_to_take = options.max_chars_per_line
                                is_last_part = (start + chars_to_take) >= token_len
                                
                                part_text = ""
                                part_mappings = {}
                                part_len = 0
                                
                                if not is_last_part:
                                    chars_to_take = options.max_chars_per_line - 1
                                    part_text = token[start : start + chars_to_take] + "-"
                                    part_len = len(part_text)
                                    for i in range(part_len - 1):
                                        chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + i)
                                        part_mappings[i] = chap_loc
                                    hyphen_map_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + chars_to_take - 1)
                                    part_mappings[part_len - 1] = hyphen_map_loc
                                else:
                                    part_text = token[start:]
                                    part_len = len(part_text)
                                    for i in range(part_len):
                                        chap_loc = CorpusLocation(chapter_index, token_start_in_chapter + start + i)
                                        part_mappings[i] = chap_loc
                                
                                current_page_data.append((part_text, part_mappings))
                                start += chars_to_take
                            
                            current_line_text = ""
                            current_line_mappings = {}

                # Advance chapter character cursor
                current_char_in_chapter += token_len
            
            # --- End of Chapter ---
            # Add the last remaining line of the chapter
            if current_line_text:
                if len(current_page_data) >= options.max_lines:
                    self._finalize_page(
                        pages, current_page_data, global_page_index,
                        page_to_chapter_map, chapter_to_page_map
                    )
                    global_page_index += 1
                    current_page_data = []
                current_page_data.append((current_line_text, current_line_mappings))

        # --- End of All Chapters ---
        # Finalize the very last page
        if current_page_data:
            self._finalize_page(
                pages, current_page_data, global_page_index,
                page_to_chapter_map, chapter_to_page_map
            )

        return PagingResult(
            pages=pages,
            pages2chaps=page_to_chapter_map,
            chaps2pages=chapter_to_page_map)

    def _finalize_page(
        self,
        pages: list[TextSegment],
        current_page_data: Sequence[tuple[str, dict[int, CorpusLocation]]],
        page_index: int,
        page_to_chapter_map: dict[CorpusLocation, CorpusLocation],
        chapter_to_page_map: dict[CorpusLocation, CorpusLocation]
    ):
        """
        Helper function to "render" a page from its line data
        and build the character maps for it.
        """
        if not current_page_data:
            return

        # 1. Create the final page text
        page_lines_text = [line_data[0] for line_data in current_page_data]
        page_content = "\n".join(page_lines_text)
        
        page_name = f"Page {page_index + 1}"
        pages.append(TextSegment(name=page_name, text=page_content))

        # 2. Build the mappings for this page
        char_in_page_counter = 0
        num_lines_on_page = len(current_page_data)
        
        # Track last valid mapping on this page for synthetic newlines
        last_known_chap_loc: CorpusLocation | None = None
        
        for i, (line_text, line_mappings) in enumerate(current_page_data):
            # Map all characters on the line
            for line_char_idx in range(len(line_text)):
                page_loc = CorpusLocation(page_index, char_in_page_counter + line_char_idx)
                
                if line_char_idx in line_mappings:
                    chap_loc = line_mappings[line_char_idx]
                    last_known_chap_loc = chap_loc # Update tracker
                    
                    # Add to forward map
                    page_to_chapter_map[page_loc] = chap_loc
                    
                    # Add to reverse map (only if not already mapped)
                    # This ensures the reverse map points to the first
                    # occurrence of a character (e.g., in 'per-')
                    if chap_loc not in chapter_to_page_map:
                        chapter_to_page_map[chap_loc] = page_loc
            
            # Advance the page character counter
            char_in_page_counter += len(line_text)
            
            # Map the newline character (if this is not the last line)
            if i < num_lines_on_page - 1:
                page_newline_loc = CorpusLocation(page_index, char_in_page_counter)
                
                # Map the newline to the last mapped character on this line.
                # If this line had no mappings (e.g., blank title line),
                # map to the last known mapping from previous lines.
                chap_loc_to_use: CorpusLocation | None = None
                
                if line_mappings:
                    # Sort keys just in case, though max() should be fine
                    last_mapped_line_idx = max(line_mappings.keys())
                    chap_loc_to_use = line_mappings[last_mapped_line_idx]
                elif last_known_chap_loc:
                    # This line is blank (or has no mappings), use the last valid mapping
                    chap_loc_to_use = last_known_chap_loc
                
                if chap_loc_to_use:
                    page_to_chapter_map[page_newline_loc] = chap_loc_to_use
                    # We don't update the reverse map for newlines,
                    # as it would overwrite the original character's mapping.
                
                # Advance counter for the newline char itself
                char_in_page_counter += 1
