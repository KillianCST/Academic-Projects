import numpy as np
import random

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def random_nucleotide_substitution(seq, substitution_rate=0.02):
    """
    Perform random nucleotide substitutions on a given DNA sequence.
    
    """
    nucleotides = 'ATGC'
    new_seq = []
    for base in seq:
        if base in nucleotides and random.random() < substitution_rate:
            # Replace base with a random nucleotide that is different.
            choices = [n for n in nucleotides if n != base]
            new_seq.append(random.choice(choices))
        else:
            new_seq.append(base)
    return ''.join(new_seq)

def augment_dataset(X_train, Y_train, reverse=True, substitute=False, substitution_rate=0.05):
    """
    Augment a dataset of DNA sequences and labels by including reverse complements and/or random substitutions.
    The original samples come first, followed by augmented samples.
    
    """
    # Convert inputs to lists for flexibility.
    raw_samples = list(X_train)
    raw_labels = list(Y_train)
    
    aug_samples = []
    aug_labels = []
    
    # Reverse complement augmentation
    if reverse:
        rev_samples = [reverse_complement(seq) for seq in X_train]
        aug_samples.extend(rev_samples)
        aug_labels.extend(Y_train)
    
    # Random substitution augmentation
    if substitute:
        sub_samples = [random_nucleotide_substitution(seq, substitution_rate) for seq in X_train]
        aug_samples.extend(sub_samples)
        aug_labels.extend(Y_train)
    
    # Concatenate original and augmented samples.
    X_aug = np.array(raw_samples + aug_samples)
    Y_aug = np.array(raw_labels + aug_labels)
    
    return X_aug, Y_aug

