from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Callable, Dict, List
from protspace.utils.prepare_json import DataProcessor
from biotrainer.input_files import BiotrainerSequenceRecord

from ..utils import get_logger
from .embedding_task import LoadEmbeddingsTask
from ..server_management import TaskInterface, TaskDTO

logger = get_logger(__name__)


class ProtSpaceTask(TaskInterface):
    def __init__(
        self,
        embedder_name: str,
        sequences: List[BiotrainerSequenceRecord],
        method: str,
        config: Dict,
    ):
        self.embedder_name = embedder_name
        self.sequences = sequences

        self.method = method
        self.config = config

    def run_task(self, update_dto_callback: Callable) -> TaskDTO:
        load_embeddings_task = LoadEmbeddingsTask(
            embedder_name=self.embedder_name,
            sequence_input=self.sequences,
            reduced=True,
            use_half_precision=False,
            device="cpu",
        )
        load_dto = None
        for dto in self.run_subtask(load_embeddings_task):
            load_dto = dto

        if not load_dto:
            return TaskDTO.failed(
                error="Loading of embeddings failed before projection!"
            )

        missing = load_dto.update["missing"]
        embeddings: List[BiotrainerSequenceRecord] = load_dto.update["embeddings"]
        if len(missing) > 0:
            return TaskDTO.failed(
                error=f"Missing number of embeddings before projection: {len(missing)}"
            )

        embedding_dict = {
            embd_record.seq_id: embd_record.embedding for embd_record in embeddings
        }
        protspace_headers = list(embedding_dict.keys())

        dimensions = self.config.pop("n_components")
        data_processor = DataProcessor(config=self.config)
        metadata = pd.DataFrame({"identifier": protspace_headers})

        logger.info(
            f"Applying {self.method.upper()} reduction. Dimensions: {dimensions}. Config: {self.config}"
        )
        reduction = data_processor.process_reduction(
            np.array(list(embedding_dict.values())),
            self.method,
            dims=dimensions,
        )
        output = data_processor.create_output(
            metadata=metadata, reductions=[reduction], headers=protspace_headers
        )
        return TaskDTO.finished(result=output)
