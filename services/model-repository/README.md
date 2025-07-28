# Model Repository

This directory contains Triton model configurations for protein embedding and prediction models. The repository includes both embedding generation models (ProtT5 and ESM-2) and downstream prediction models for various protein properties.

## Model Categories

### 🧬 Embedding Models
Generate protein sequence embeddings for downstream tasks.

### 🔬 Per-Residue Prediction Models  
Predict properties for each amino acid position in the sequence.

### 🎯 Sequence-Level Prediction Models
Predict global properties for the entire protein sequence.

### ⚙️ Ensemble Pipelines
Complete prediction workflows including preprocessing and postprocessing.

---

## Embedding Models

### ProtT5 Models
- **prot_t5_pipeline**: Main ensemble pipeline (1024-dim embeddings)
  - Uses `_internal_prott5_tokenizer` (T5-based with space-separated amino acids)
  - Uses `_internal_prott5_onnx` (ProtT5 transformer model)

### ESM-2 Models
- **esm2_t33_pipeline**: ESM-2 T33 650M ensemble (1280-dim embeddings)
- **esm2_t36_pipeline**: ESM-2 T36 3B ensemble (2560-dim embeddings)
  - Both use `_internal_esm2_tokenizer` (ESM-2-based, no spaces needed)
  - Use respective `_internal_esm2_t33_onnx` / `_internal_esm2_t36_onnx` models

---

## Per-Residue Prediction Models

All per-residue models use ProtT5 embeddings (1024-dim) as input and predict properties for each amino acid position.

### Secondary Structure (`prott5_sec/`)
- **Model Name**: `secondary_structure_model`
- **Purpose**: Predict protein secondary structure
- **Input**: ProtT5 embeddings (FP32, [-1, 1024])
- **Output**: 3-class predictions (FP32, [-1, 3])
  - Classes: `H` (Helix), `E` (Beta-sheet), `C` (Coil)
- **Architecture**: ONNX neural network
- **Batch Size**: Up to 16 sequences
- **Hardware**: GPU inference

### Conservation (`prott5_cons/`)
- **Model Name**: `conservation_model`
- **Purpose**: Predict evolutionary conservation scores
- **Input**: ProtT5 embeddings (FP32, [-1, 1024])
- **Output**: 9-class conservation scores (FP32, [-1, 9])
  - Classes: Conservation levels from highly variable to highly conserved
- **Architecture**: ONNX neural network
- **Batch Size**: Up to 16 sequences  
- **Hardware**: GPU inference

### Binding Sites (`bind_embed/`)
- **Model Name**: `binding_sites_model`
- **Purpose**: Predict protein-ligand binding sites
- **Input**: ProtT5 embeddings (FP32, [-1, 1024])
- **Output**: Binary classification (FP32, [-1, 2])
  - Classes: `Non-binding` vs `Binding`
- **Architecture**: ONNX neural network
- **Batch Size**: Up to 16 sequences
- **Hardware**: GPU inference

### Intrinsic Disorder (SETH)

#### Simple Model (`seth/`)
- **Model Name**: `disorder_onnx`
- **Purpose**: Raw disorder prediction scores
- **Input**: ProtT5 embeddings (FP32, [-1, 1024])
- **Output**: Binary classification scores (FP32, [-1, 2])
  - Classes: `Ordered` vs `Disordered`
- **Architecture**: ONNX neural network
- **Hardware**: GPU inference

#### Complete Pipeline (`seth_pipeline/`)
- **Model Name**: `disorder_ensemble`
- **Purpose**: End-to-end disorder prediction with postprocessing
- **Input**: ProtT5 embeddings (FP32, [-1, 1024])
- **Output**: Per-residue string predictions (STRING, [-1])
  - Format: Concatenated per-residue labels ("OrderedOrderedDisordered...")
- **Architecture**: Ensemble (seth → seth_postprocess)
- **Components**:
  1. `seth/`: Raw ONNX predictions (model name: `disorder_onnx`)
  2. `seth_postprocess/`: Python postprocessing (model name: `disorder_postprocess`)

---

## Sequence-Level Prediction Models

These models predict global properties for entire protein sequences using pooled embeddings.

### Membrane Localization (`tmbed/`)
- **Model Name**: `membrane_localization_model`
- **Purpose**: Predict membrane association and topology
- **Input**: ProtT5 embeddings (FP32, [-1, 1024]) - will be pooled internally
- **Output**: 4-class prediction (FP32, [4])
  - Classes: Membrane topology categories
- **Architecture**: ONNX neural network with sequence pooling
- **Batch Size**: Up to 16 sequences
- **Hardware**: GPU inference

### Subcellular Localization (`light_attention_subcell/`)
- **Model Name**: `subcellular_localization_model`
- **Purpose**: Predict subcellular localization
- **Input**: ProtT5 embeddings (FP32, [-1, 1024]) - will be pooled internally  
- **Output**: 10-class prediction (FP32, [10])
  - Classes: Organelles and cellular compartments
- **Architecture**: ONNX neural network with sequence pooling
- **Batch Size**: Up to 16 sequences
- **Hardware**: GPU inference

### Variant Effect Prediction (`vespag/`)
- **Model Name**: `vespag` (VESPA-G)
- **Purpose**: Predict the functional impact of protein variants
- **Input**: Likely protein embeddings or variant-specific features
- **Output**: Variant effect scores
- **Architecture**: FNN ONNX model (`fnn.onnx`)
- **Status**: 🚧 Configuration pending
- **Hardware**: TBD

---

## Directory Structure

```
services/model-repository/
├── Embedding Models
│   ├── prot_t5_pipeline/           # ProtT5 ensemble (1024-dim)
│   ├── esm2_t33_pipeline/          # ESM-2 T33 ensemble (1280-dim)
│   ├── esm2_t36_pipeline/          # ESM-2 T36 ensemble (2560-dim)
│   ├── _internal_prott5_tokenizer/ # ProtT5 tokenizer
│   ├── _internal_prott5_onnx/      # ProtT5 ONNX model
│   ├── _internal_esm2_tokenizer/   # ESM-2 tokenizer (shared)
│   ├── _internal_esm2_t33_onnx/    # ESM-2 T33 ONNX model
│   └── _internal_esm2_t36_onnx/    # ESM-2 T36 ONNX model
├── Per-Residue Predictions
│   ├── prott5_sec/                 # Secondary structure
│   ├── prott5_cons/                # Conservation scores
│   ├── bind_embed/                 # Binding sites
│   ├── seth/                       # Disorder (raw)
│   ├── seth_pipeline/              # Disorder (complete)
│   └── seth_postprocess/           # Disorder postprocessing
├── Sequence-Level Predictions
│   ├── tmbed/                      # Membrane localization
│   ├── light_attention_subcell/    # Subcellular localization
│   └── vespag/                     # Variant effect prediction
└── README.md                       # This documentation
```

---

## Model Configurations

### Performance Settings
- **Embedding Models**: Batch size 1-8, optimized for throughput
- **Prediction Models**: Batch size up to 16, optimized for inference speed
- **Dynamic Batching**: Automatic batching with preferred sizes
- **Queue Delays**: 5-10ms max for responsive predictions

### Hardware Requirements
- **CPU Models**: Tokenizers, postprocessing
- **GPU Models**: All ONNX inference models (embedding + prediction)
- **Memory**: Variable based on sequence length and batch size

### Data Types
- **Embeddings**: FP16 (storage) / FP32 (computation)
- **Predictions**: FP32 (scores) / STRING (labels)
- **Tokenization**: INT64 (token IDs)

### Naming Convention
- **Directory Names**: Descriptive names reflecting the actual model/method
- **Model Names**: Internal Triton model names (may differ from directory names)
- **Example**: Directory `prott5_sec/` contains model named `secondary_structure_model`

---

## Supported Embedding Dimensions

| Model | Embedding Dimension | Compatible Predictions |
|-------|-------------------|----------------------|
| ProtT5 | 1024 | All current prediction models |
| ESM-2 T33 650M | 1280 | Future prediction models |
| ESM-2 T36 3B | 2560 | Future prediction models |

> **Note**: Current prediction models are trained on ProtT5 embeddings (1024-dim). Future versions will support ESM-2 embeddings.

---

## Usage Patterns

### 1. **Embedding Generation**
```
Input: Protein sequences → Tokenizer → ONNX Model → Embeddings
```

### 2. **Per-Residue Prediction**
```
Embeddings → Prediction Model → Per-residue scores/labels
```

### 3. **Sequence-Level Prediction**
```
Embeddings → Pooling + Prediction Model → Sequence-level scores
```

### 4. **Complete Ensemble**
```
Embeddings → ONNX Model → Postprocessing → Final predictions
```

---

## Cache Integration

The embedding service automatically caches results using model-specific keys:
- `protein_emb:prot_t5:sequence_hash`
- `protein_emb:esm2_t33_650M:sequence_hash`
- `protein_emb:esm2_t36_3B:sequence_hash`

Prediction results can be cached separately by downstream services.

---

## Deployment Requirements

### Embedding Models
To deploy the embedding models, provide the actual model files:

#### ESM-2 Tokenizer
- Uses HuggingFace `facebook/esm2_t33_650M_UR50D` tokenizer
- The model.py automatically downloads from HuggingFace during initialization

#### ESM-2 ONNX Models
Replace placeholder files with actual ONNX models:

**For T33 650M (`_internal_esm2_t33_onnx/1/model.onnx`):**
- Input: `input_ids` and `attention_mask` (INT64)
- Output: `last_hidden_state` (FP16, shape: [-1, 1280])

**For T36 3B (`_internal_esm2_t36_onnx/1/model.onnx`):**
- Input: `input_ids` and `attention_mask` (INT64)
- Output: `last_hidden_state` (FP16, shape: [-1, 2560])

### Prediction Models
All prediction models require:
1. Corresponding ONNX model files in `1/model.onnx`
2. GPU compute capability for optimal performance
3. ProtT5 embeddings as input (1024-dimensional)

### Model Dependencies
```
Protein Sequence
    ↓
ProtT5 Pipeline (embedding generation)
    ↓
[Multiple prediction models can run in parallel]
    ↓
Final predictions
```

### Special Requirements
- **VESPAG**: Requires configuration file (`config.pbtxt`) to be created
- **SETH Pipeline**: Requires coordination between `seth` and `seth_postprocess` models 