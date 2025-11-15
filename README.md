# audiobookutils

# Operations
## Align EPUB and audio files
Given a book in EPUB format and audio file readings for that book, align them, and output an SRT subtitle file for each audio file, where the subtitle contains text from the book highlighted on reading of each word.

1. Create a complete transcription given audiobook readings using [whisperX](https://github.com/m-bain/whisperX) with per-word timings.
```python
uv run -m audiobookutils.preprocessing \
	audio_files=["data/books/The Sorrows of Satan/part1.mp3","data/books/The Sorrows of Satan/part2.mp3","data/books/The Sorrows of Satan/part3.mp3"] \
	result_dir=out/results/sorrowsofsatan
```
See [./conf/preprocessing.yaml](./conf/preprocessing.yaml) for options.

This will generate `out/results/sorrowsofsatan/transcription.json`.

2. Align generated transcriptions and book (EPUB), then generate SRT subtitle file(s) for each audio book reading.
```
uv run -m audiobookutils.engine \
  output_dir="out/results/sorrowsofsatan" \
  ebook_path="data/books/The Sorrows of Satan/corelli-sorrows-of-satan.epub" \
  transcription_result_paths=["out/results/sorrowsofsatan/transcription.json"] \
  paging.max_lines=17 \
  paging.max_characters_per_line=75
```

See [./conf/match_book_and_audio.yaml](./conf/match_book_and_audio.yaml) and [./conf/paging/paging_base.yaml](./conf/paging/paging_base.yaml) for more options; the latter configuration controls the virtual *page* that will be displayed as a subtitle (note that some video players may have limitations as to how much subtitle can be displayed at a single point).

The alignment process is somewhat expensive, and will produce `ebook_and_audio_transcriptions_match.json` in your specified output directory. If you run `audiobookutils.engine` again while this file exists, previous alignment data will be loaded from this file instead of rerunning the algorithm.

# Installation
```sh
uv sync
```

## Troubleshooting
### Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
- https://github.com/m-bain/whisperX/issues/902

Check that the CuDNN modules are installed for this project's environment under `.venv/lib/python3.11/site-packages/nvidia/cudnn/lib` (you may need to install `nvidia-cudnn` or an equivalent package on the host system before installing the python dependences).
Then set environment variable:
```sh
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
```

# Dev
## Roadmap
- Automatic audiobook reading generation via TTS

## Notes
- Due to the way the forced aligner works, during fuzzy matching the audio transcriptions have an extra space in between words with the default setting (`append_space_to_words=true` in [./conf/preprocessing.yaml](./conf/preprocessing.yaml); these "words" are joined for fuzzy matching with a space delimiter creating double space), to prevent multiple words from being matched simultaneosly. This was initially an unintentional bug during development, but in the end seems to work out, despite the hackiness of the solution.

## Related Project
There are multiple open source projects that achieve a similar goal to this project, which I was not aware of when I first started development. This project is still unique in that it produces subtitle files for each audiobook reading, but practically, preexisting tools - especially ones that utilize the media overlay feature of EPUB 3 to create *read aloud* books are superior methods of consuming/storing audiobooks.

- [aeneas](https://github.com/readbeyond/aeneas) - *a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment)*.
- [syncabook](https://github.com/r4victor/syncabook) - *a set of tools for creating ebooks with synchronized text and audio (a.k.a. read along, read aloud; like Amazon's Whispersync)*
- [Storyteller](https://gitlab.com/storyteller-platform/storyteller) - Complete solution to align audiobook readings with underlying text for consumption on mobile devices

## References
- [StackOverflow - how to parse text from each chapter in epub?](https://stackoverflow.com/questions/56410564/how-to-parse-text-from-each-chapter-in-epub)

