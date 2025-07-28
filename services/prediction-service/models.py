"""
Model configurations and metadata for prediction service.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pydantic import BaseModel


class PredictionType(str, Enum):
    """Type of prediction."""
    PER_RESIDUE = "per_residue"
    PER_SEQUENCE = "per_sequence"


class ValueType(str, Enum):
    """Type of output values."""
    CLASS = "class"
    FLOAT = "float"
    INTEGER = "integer"


class OutputClass(BaseModel):
    """Class definition for categorical outputs."""
    label: str
    description: str


class ModelOutput(BaseModel):
    """Output specification for a model."""
    name: str
    description: str
    output_type: PredictionType
    value_type: ValueType
    classes: Optional[Dict[str, OutputClass]] = None  # for categorical outputs
    value_range: Optional[Tuple[float, float]] = None  # for continuous outputs (min, max)
    unit: Optional[str] = None  # optional, for numerical outputs


class ModelMetadata(BaseModel):
    """Metadata for a prediction model."""
    name: str
    triton_name: str
    description: str
    authors: str
    citation: Optional[str] = None
    license: Optional[str] = None
    outputs: List[ModelOutput]
    model_size: str  # e.g., "926.7 KB"
    testset_performance: Optional[str] = None
    training_data_link: Optional[str] = None
    embedder: str  # e.g., "Rostlab/prot_t5_xl_uniref50"
    input_shape: List[int]  # Expected input shape (excluding batch)
    requires_mask: bool = False
    requires_ensemble: bool = False
    
    # Computed properties for backward compatibility
    @property
    def output_type(self) -> PredictionType:
        """Primary output type (for backward compatibility)."""
        return self.outputs[0].output_type if self.outputs else PredictionType.PER_RESIDUE
    
    @property
    def output_classes(self) -> int:
        """Number of output classes (for backward compatibility)."""
        first_output = self.outputs[0] if self.outputs else None
        if first_output and first_output.classes:
            return len(first_output.classes)
        elif first_output and first_output.value_range:
            return 1  # Continuous output
        return 0
    
    @property
    def class_labels(self) -> Dict[int, str]:
        """Class labels (for backward compatibility)."""
        first_output = self.outputs[0] if self.outputs else None
        if first_output and first_output.classes:
            return {i: cls.label for i, cls in enumerate(first_output.classes.values())}
        return {}
    
    @property
    def model_size_mb(self) -> Optional[float]:
        """Model size in MB (computed from model_size string)."""
        if not self.model_size:
            return None
        try:
            # Parse strings like "926.7 KB", "1.4 MB", etc.
            parts = self.model_size.split()
            if len(parts) == 2:
                value, unit = parts
                value = float(value)
                if unit.upper() == "KB":
                    return value / 1024
                elif unit.upper() == "MB":
                    return value
                elif unit.upper() == "GB":
                    return value * 1024
        except (ValueError, IndexError):
            pass
        return None


# Model configurations registry
MODELS_CONFIG: Dict[str, ModelMetadata] = {
    "ProtT5Conservation": ModelMetadata(
        name="ProtT5Conservation",
        triton_name="conservation_model",
        description="VESPA model for protein residue conservation prediction",
        authors="Céline Marquet and Michael Heinzinger and Tobias Olenyi and Christian Dallago and Kyra Erckert and Michael Bernhofer and Dmitrii Nechaev and Burkhard Rost",
        citation="https://doi.org/10.1007/s00439-021-02411-y",
        license="AGPL-3.0",
        outputs=[
            ModelOutput(
                name="conservation",
                description="Per-residue evolutionary conservation prediction, as defined by 10.1093/bioinformatics/bth070",
                output_type=PredictionType.PER_RESIDUE,
                value_type=ValueType.CLASS,
                classes={
                    "0": OutputClass(label="Variable", description="Residue is evolutionarily variable"),
                    "1": OutputClass(label="Variable", description="Residue is evolutionarily variable"),
                    "2": OutputClass(label="Variable", description="Residue is evolutionarily variable"),
                    "3": OutputClass(label="Variable", description="Residue is evolutionarily variable"),
                    "4": OutputClass(label="Average", description="Residue is equally conserved and variable"),
                    "5": OutputClass(label="Average", description="Residue is equally conserved and variable"),
                    "6": OutputClass(label="Average", description="Residue is equally conserved and variable"),
                    "7": OutputClass(label="Conserved", description="Residue is evolutionarily conserved"),
                    "8": OutputClass(label="Conserved", description="Residue is evolutionarily conserved"),
                }
            )
        ],
        model_size="926.7 KB",
        testset_performance="",
        training_data_link="http://data.bioembeddings.com/public/design/",
        embedder="Rostlab/prot_t5_xl_uniref50",
        input_shape=[-1, 1024],
    ),
    
    "ProtT5SecondaryStructure": ModelMetadata(
        name="ProtT5SecondaryStructure",
        triton_name="secondary_structure_model",
        description="ProtT5-based secondary structure prediction",
        authors="Michael Heinzinger and Ahmed Elnaggar and Yu Wang and Christian Dallago and Dmitrii Nechaev and Florian Matthes and Burkhard Rost",
        citation="https://doi.org/10.1101/2020.07.12.199554",
        license="MIT",
        output_type=PredictionType.PER_RESIDUE,
        output_classes=3,
        class_labels={0: "H", 1: "E", 2: "C"},  # Helix, Beta sheet, Coil
        input_shape=[-1, 1024],
        output_shape=[-1, 3],
        model_size_mb=1.2
    ),
    
    "SETH": ModelMetadata(
        name="SETH",
        triton_name="disorder_ensemble",
        description="Prediction of intrinsically disordered regions",
        authors="Hannes Stärk and Christian Dallago and Michael Heinzinger and Burkhard Rost",
        citation="https://doi.org/10.1093/bioinformatics/bty1007",
        license="MIT",
        outputs=[
            ModelOutput(
                name="disorder",
                description="Disorder scores: Below 8 - disorder, Above 8 - order, as defined by CheZOD Z-scores: https://doi.org/10.1007/978-1-0716-0524-0_15",
                output_type=PredictionType.PER_RESIDUE,
                value_type=ValueType.FLOAT,
                value_range=(0.0, 16.0),  # CheZOD Z-scores range
                unit="z-score"
            )
        ],
        model_size="575.1 KB",
        testset_performance="",
        training_data_link="http://data.bioembeddings.com/public/design/",
        embedder="Rostlab/prot_t5_xl_uniref50",
        input_shape=[-1, 1024],
    ),
    
    "BindEmbed": ModelMetadata(
        name="BindEmbed",
        triton_name="binding_sites_model",
        description="Prediction of protein binding sites",
        authors="Konstantin Schütze and Michael Heinzinger and Martin Steinegger and Burkhard Rost",
        citation="https://doi.org/10.1101/2022.03.23.485545",
        license="MIT",
        output_type=PredictionType.PER_RESIDUE,
        output_classes=2,
        class_labels={0: "Non-binding", 1: "Binding"},
        input_shape=[-1, 1024],
        output_shape=[-1],
        model_size_mb=1.8
    ),
    
    "LightAttentionMembrane": ModelMetadata(
        name="LightAttentionMembrane",
        triton_name="membrane_localization_model",
        description="Membrane protein localization prediction",
        authors="Hannes Stärk and Christian Dallago and Michael Heinzinger and Burkhard Rost",
        license="MIT",
        output_type="per_sequence",
        output_classes=4,
        class_labels={0: "Cytoplasm", 1: "Inner_Membrane", 2: "Outer_Membrane", 3: "Extracellular"},
        input_shape=[-1, 1024],
        output_shape=[4],
        model_size_mb=3.2
    ),
    
    "LightAttentionSubcellularLocalization": ModelMetadata(
        name="LightAttentionSubcellularLocalization",
        triton_name="subcellular_localization_model",
        description="Subcellular localization prediction",
        authors="Hannes Stärk and Christian Dallago and Michael Heinzinger and Burkhard Rost",
        license="MIT",
        output_type=PredictionType.PER_SEQUENCE,
        output_classes=10,
        class_labels={
            0: "Cytoplasm", 1: "Nucleus", 2: "Extracelluar", 3: "Mitochondrion",
            4: "Cell_membrane", 5: "Endoplasmic_reticulum", 6: "Plastid",
            7: "Golgi_apparatus", 8: "Lysosome", 9: "Peroxisome"
        },
        input_shape=[-1, 1024],
        output_shape=[10],
        model_size_mb=4.1
    ),
    
    # Phase 3: Complex models
    "TMbed": ModelMetadata(
        name="TMbed",
        triton_name="tmbed_ensemble",
        description="Prediction of transmembrane proteins with topology",
        authors="Michael Bernhofer and Burkhard Rost",
        citation="https://doi.org/10.1186/s12859-022-04873-x",
        license="Apache-2.0",
        output_type=PredictionType.PER_RESIDUE,
        output_classes=7,
        class_labels={
            0: "B", 1: "b",  # Transmembrane beta strand (IN->OUT, OUT->IN)
            2: "H", 3: "h",  # Transmembrane alpha helix (IN->OUT, OUT->IN)
            4: "S",          # Signal peptide
            5: "i", 6: "o"   # Non-transmembrane (inside, outside)
        },
        input_shape=[-1, 1024],
        output_shape=[-1, 7],
        requires_mask=True,
        requires_ensemble=True,
        model_size_mb=15.8
    ),
    
    "VespaG": ModelMetadata(
        name="VespaG",
        triton_name="variant_effects_model",
        description="Variant effect prediction on protein function",
        authors="Christian Dallago and Jody Mou and Kadina E. Johnston and Bruce J. Wittmann and Nicholas Bhattacharya and Samuel Goldman and Ali Madani and Kevin K. Yang",
        citation="https://doi.org/10.1038/s41588-023-01465-0",
        license="MIT",
        outputs=[
            ModelOutput(
                name="variant_effect",
                description="Per-residue variant effect scores for protein function prediction",
                output_type=PredictionType.PER_RESIDUE,
                value_type=ValueType.FLOAT,
                value_range=(0.0, 1.0),  # Scores are normalized between 0 and 1
                unit="score"
            )
        ],
        model_size="2.6 MB",
        testset_performance="",
        training_data_link="https://zenodo.org/records/11085958",
        embedder="facebook/esm2_t36_3B_UR50D",
        input_shape=[-1, 1024],
    )
}


def get_model_metadata(model_name: str) -> Optional[ModelMetadata]:
    """Get metadata for a specific model."""
    return MODELS_CONFIG.get(model_name)


def get_available_models() -> List[str]:
    """Get list of all available model names."""
    return list(MODELS_CONFIG.keys())


def get_all_models_metadata() -> Dict[str, ModelMetadata]:
    """Get metadata for all models."""
    return MODELS_CONFIG.copy()