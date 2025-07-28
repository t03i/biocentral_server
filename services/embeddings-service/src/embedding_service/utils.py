"""
Utility functions for the Protein Embedding Service
"""
from typing import List
from .config import config


def validate_protein_sequences(sequences: List[str]) -> List[str]:
    """Validate and clean protein sequences"""
    if not sequences:
        raise ValueError("At least one sequence is required")
    if len(sequences) > config.max_sequences_per_request:
        raise ValueError(f"Maximum {config.max_sequences_per_request} sequences per request")
    
    cleaned_sequences = []
    for i, seq in enumerate(sequences):
        if not seq or not isinstance(seq, str):
            raise ValueError(f"Sequence {i} is empty or not a string")
        
        # Clean and validate sequence
        cleaned_seq = seq.upper().replace('*', '').strip()
        
        if len(cleaned_seq) < config.min_sequence_length:
            raise ValueError(f"Sequence {i} too short (min {config.min_sequence_length} residues), got {len(cleaned_seq)}")
        if len(cleaned_seq) > config.max_sequence_length:
            raise ValueError(f"Sequence {i} too long (max {config.max_sequence_length} residues), got {len(cleaned_seq)}")
        
        # Basic protein sequence validation
        valid_chars = set('ACDEFGHIKLMNPQRSTVWYXU-')
        if not set(cleaned_seq).issubset(valid_chars):
            invalid_chars = set(cleaned_seq) - valid_chars
            raise ValueError(f"Sequence {i} contains invalid characters: {invalid_chars}")
        
        cleaned_sequences.append(cleaned_seq)
    
    return cleaned_sequences 