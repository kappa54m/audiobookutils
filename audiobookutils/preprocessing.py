from .logging import LoggerFactory
from .common import AudioTranscriptionChunk, AudioTranscription
 
from abc import ABC, abstractmethod
from typing import Sequence
import os
from pathlib import Path
import json

import hydra
from omegaconf import DictConfig
import torch
import whisperx


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self) -> AudioTranscription:
        pass


class AudioPreprocessor(Preprocessor):
    """
    Preprocessor for audiobook readings (audio files only).
    Utilizes WhisperX (https://github.com/m-bain/whisperx) for transcription.
    """
    def __init__(self, loggerfactory: LoggerFactory, file_paths: Sequence[os.PathLike],
                 device: str, append_space_to_words: bool=True,
                 batch_size=16, compute_type='float16', whisper_model='large-v2'):
        """
        Args:
            file_paths: Paths to audio files for the audiobook
            append_space_to_words: For each audio chunk, append space at the end
        """
        self.logger = loggerfactory.get_logger(self.__class__.__name__)
        self.file_paths = file_paths
        self.append_space_to_words = append_space_to_words
        for pth in self.file_paths:
            if not Path(pth).is_file():
                raise ValueError("Invalid file: {}".format(pth))

        self.device = device
        self.logger.info("Device: %s", device)
        if isinstance(whisper_model, str):
            self.logger.info("Loading whisper model '%s' on device %s (compute type: %s)",
                             whisper_model, device, compute_type)
            self.model = whisperx.load_model(whisper_model, self.device, compute_type=compute_type)
        else:
            self.model = whisper_model.to(self.device)
        self.batch_size = batch_size

    def preprocess(self) -> AudioTranscription:
        transcription_chunks = []
        for ifile, audio_file_path in enumerate(self.file_paths):
            self.logger.info("Transcribing file %d/%d: %s", ifile+1, len(self.file_paths), audio_file_path)
            audio = whisperx.load_audio(audio_file_path)
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            # Perform alignment
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            align_result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            for segment in align_result['segments']:
                for word_seg in segment['words']:
                    word = word_seg['word']
                    if self.append_space_to_words:
                        word += " "
                    transcription_chunks.append(AudioTranscriptionChunk(
                        file_path=audio_file_path,
                        start_secs=float(word_seg['start']),
                        end_secs=float(word_seg['end']),
                        text=word))
        return AudioTranscription(chunks=transcription_chunks)


@hydra.main(version_base=None, config_path='../conf', config_name='preprocessing')
def main(conf: DictConfig):
    logging_conf = conf['logging']

    lf = LoggerFactory(console_logging_level=logging_conf['console_logging_level'],
                       do_file_logging=logging_conf['do_file_logging'],
                       file_logging_level=logging_conf['file_logging_level'],
                       file_logging_dir=logging_conf['file_logging_dir'])
    logger = lf.get_logger(__name__)

    audio_file_paths = conf['audio_files']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = AudioPreprocessor(loggerfactory=lf,
                                     file_paths=audio_file_paths,
                                     device=device,
                                     batch_size=conf['batch_size'], compute_type=conf['compute_type'],
                                     whisper_model=conf['transcription_model'])
    logger.info("Preprocessing...")
    transcription_result = preprocessor.preprocess()
    result_dir = Path(conf['result_dir'])
    result_dir.mkdir(exist_ok=True, parents=True)
    transcription_result_path = result_dir / "transcription.json"
    logger.info("Saving transcription results to: %s", transcription_result_path)
    transcription_result_as_dict = transcription_result.to_serializable_dict()
    with open(transcription_result_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_result_as_dict, f, ensure_ascii=False, indent=2)
    logger.info("Preprocessing finished")

if __name__ == '__main__':
    main()
