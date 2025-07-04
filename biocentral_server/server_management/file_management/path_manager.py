from pathlib import Path
from typing import Final, List, Optional, Tuple

from .storage_file_type import StorageFileType


class PathManager:
    base_path: Final[Path] = Path("")

    _fasta_files_path: Final[Path] = Path("fasta_files/")
    _embeddings_files_path: Final[Path] = Path("embeddings/")
    _models_files_path: Final[Path] = Path("models/")
    _external_models_path: Final[Path] = Path("external_models/")
    _subdirectories_hash_dir: Final[List[Path]] = [
        _fasta_files_path,
        _embeddings_files_path,
        _models_files_path,
        _external_models_path,
    ]

    def __init__(self, user_id: str):
        self.user_id = user_id

    def _base_user_path(self) -> Path:
        return self.base_path / self.user_id

    def get_database_path(self, database_hash: str) -> Path:
        return self._base_user_path() / database_hash

    def get_embeddings_files_path(self, database_hash: str) -> Path:
        return self.get_database_path(database_hash) / self._embeddings_files_path

    def _get_fasta_file_path(self, database_hash: str) -> Path:
        return self.get_database_path(database_hash) / self._fasta_files_path

    def _get_models_files_path(self) -> Path:
        return self._base_user_path() / self._models_files_path

    def _get_external_models_path(self) -> Path:
        return self._base_user_path() / self._external_models_path

    def get_biotrainer_model_path(self, model_hash: str) -> Path:
        return self._get_models_files_path() / model_hash

    @staticmethod
    def _storage_file_type_to_file_name(
        file_type: StorageFileType, embedder_name: Optional[str] = ""
    ) -> str:
        return {
            StorageFileType.INPUT: "input_file.fasta",
            StorageFileType.BIOTRAINER_CONFIG: "config_file.yaml",
            StorageFileType.BIOTRAINER_LOGGING: "logger_out.log",
            StorageFileType.BIOTRAINER_RESULT: "out.yml",
            StorageFileType.ONNX_MODEL: f"{embedder_name}.onnx",
            StorageFileType.TOKENIZER_CONFIG: f"{embedder_name}_tokenizer_config.json",
        }[file_type]

    def _storage_file_type_to_path(
        self,
        file_type: StorageFileType,
        database_hash: Optional[str] = "",
        model_hash: Optional[str] = "",
    ) -> Path:
        return {
            StorageFileType.INPUT: self._get_fasta_file_path(database_hash),
            StorageFileType.BIOTRAINER_CONFIG: self._get_models_files_path()
            / Path(model_hash),
            StorageFileType.BIOTRAINER_LOGGING: self._get_models_files_path()
            / Path(model_hash),
            StorageFileType.BIOTRAINER_RESULT: self._get_models_files_path()
            / Path(model_hash),
            StorageFileType.BIOTRAINER_CHECKPOINT: self._get_models_files_path()
            / Path(model_hash),
            # Gets searched for pt file(s)
            StorageFileType.ONNX_MODEL: self._get_external_models_path(),
            StorageFileType.TOKENIZER_CONFIG: self._get_external_models_path(),
        }[file_type]

    def get_file_name_and_path(
        self,
        file_type: StorageFileType,
        database_hash: Optional[str] = "",
        model_hash: Optional[str] = "",
        embedder_name: Optional[str] = "",
    ) -> Tuple[str, Path]:
        return (
            self._storage_file_type_to_file_name(
                file_type=file_type, embedder_name=embedder_name
            ),
            self._storage_file_type_to_path(
                database_hash=database_hash, file_type=file_type, model_hash=model_hash
            ),
        )
