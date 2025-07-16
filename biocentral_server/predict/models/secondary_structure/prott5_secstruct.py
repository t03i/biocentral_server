import torch
import numpy as np

from typing import Dict
from biotrainer.protocols import Protocol

from ..base_model import BaseModel, ModelMetadata, ModelOutput, OutputClass, OutputType


class ProtT5SecondaryStructure(BaseModel):
    def __init__(self, batch_size):
        super().__init__(
            batch_size=batch_size,
            uses_ensemble=False,
            requires_mask=False,
            requires_transpose=False,
        )
        self.label_mapping_3_states = {0: "H", 1: "E", 2: "L"}
        self.label_mapping_8_states = {
            idx: state for idx, state in enumerate("GHIBESTC")
        }

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ProtT5SecondaryStructure",
            protocol=Protocol.residue_to_class,
            description="ProtT5 secondary structure prediction",
            authors="Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Yu, Wang and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard",
            model_link="https://github.com/agemagician/ProtTrans",
            citation="https://doi.org/10.1109/TPAMI.2021.3095381",
            licence="MIT",
            outputs=[
                ModelOutput(
                    name="d3_Yhat",
                    description="3-state secondary structure prediction",
                    output_type=OutputType.PER_RESIDUE,
                    value_type=str,
                    classes={
                        "H": OutputClass(
                            label="Helix",
                            description="Residue is part of an alpha helix",
                        ),
                        "E": OutputClass(
                            label="Sheet", description="Residue is part of a beta sheet"
                        ),
                        "L": OutputClass(
                            label="Other",
                            description="Residue is part of a loop or coil",
                        ),
                    },
                ),
                ModelOutput(
                    name="d8_Yhat",
                    description="8-state DSSP secondary structure prediction",
                    output_type=OutputType.PER_RESIDUE,
                    value_type=str,
                    classes={
                        "G": OutputClass(
                            label="3-10 Helix",
                            description="Residue is part of a 3-10 helix",
                        ),
                        "H": OutputClass(
                            label="Alpha Helix",
                            description="Residue is part of an alpha helix",
                        ),
                        "I": OutputClass(
                            label="Pi Helix",
                            description="Residue is part of a pi helix",
                        ),
                        "B": OutputClass(
                            label="Beta Bridge",
                            description="Residue is part of an isolated beta bridge",
                        ),
                        "E": OutputClass(
                            label="Extended Strand",
                            description="Residue is part of an extended strand in a beta ladder",
                        ),
                        "S": OutputClass(
                            label="Bend", description="Residue is part of a bend"
                        ),
                        "T": OutputClass(
                            label="Turn",
                            description="Residue is part of a hydrogen-bonded turn",
                        ),
                        "C": OutputClass(
                            label="Coil",
                            description="Residue is part of a coil (none of the above)",
                        ),
                    },
                ),
            ],
            model_size="929.0 KB",
            testset_performance="",
            training_data_link="http://data.bioembeddings.com/public/design/",
            embedder="Rostlab/prot_t5_xl_uniref50",
        )

    def predict(self, sequences: Dict[str, str], embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = list(embeddings.keys())
        model_output = {"d3_Yhat": [], "d8_Yhat": []}
        for batch in inputs:
            d3_Yhat, d8_Yhat = self.model.run(None, batch)
            d3_Yhat = torch.from_numpy(np.float32(np.stack(d3_Yhat)))
            d8_Yhat = torch.from_numpy(np.float32(np.stack(d8_Yhat)))
            d3_Yhat = self._finalize_raw_prediction(
                torch.max(d3_Yhat, dim=-1, keepdim=True)[1], dtype=np.byte
            )
            d8_Yhat = self._finalize_raw_prediction(
                torch.max(d8_Yhat, dim=-1, keepdim=True)[1], dtype=np.byte
            )
            model_output["d3_Yhat"].extend(d3_Yhat)
            model_output["d8_Yhat"].extend(d8_Yhat)
        return self._post_process(
            model_output=model_output,
            embedding_ids=embedding_ids,
            label_maps={
                "d3_Yhat": self.label_mapping_3_states,
                "d8_Yhat": self.label_mapping_8_states,
            },
        )
