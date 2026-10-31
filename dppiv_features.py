"""Sequence descriptors and chemical fingerprints for DPP-IV peptides.

CTD is the 273-dimensional iFeature encoding: 13 properties, each with
3 composition, 3 transition and 15 distribution coordinates. PAAC and
APAAC use Chou's lambda=1, weight=0.05 forms, so the dimensions are 21
and 22. AAI is the mean of 11 physicochemical scales. Fingerprints are
EState (79), MACCS (166), PubChem (881) and OpenBabel FP4 (307), the
last of these via the Laggner SMARTS set.
"""

from __future__ import annotations

import numpy as np

from dppiv_logit import AA, AA_INDEX, N_AA

# iFeature CTD groups. Each property's three strings partition the 20 letters.
_CTD_PROPERTIES = (
    "hydrophobicity_PRAM900101",
    "hydrophobicity_ARGP820101",
    "hydrophobicity_ZIMJ680101",
    "hydrophobicity_PONP930101",
    "hydrophobicity_CASG920101",
    "hydrophobicity_ENGD860101",
    "hydrophobicity_FASG890101",
    "normwaalsvolume",
    "polarity",
    "polarizability",
    "charge",
    "secondarystruct",
    "solventaccess",
)
_CTD_GROUP1 = {
    "hydrophobicity_PRAM900101": "RKEDQN",
    "hydrophobicity_ARGP820101": "QSTNGDE",
    "hydrophobicity_ZIMJ680101": "QNGSWTDERA",
    "hydrophobicity_PONP930101": "KPDESNQT",
    "hydrophobicity_CASG920101": "KDEQPSRNTG",
    "hydrophobicity_ENGD860101": "RDKENQHYP",
    "hydrophobicity_FASG890101": "KERSQD",
    "normwaalsvolume": "GASTPDC",
    "polarity": "LIFWCMVY",
    "polarizability": "GASDT",
    "charge": "KR",
    "secondarystruct": "EALMQKRH",
    "solventaccess": "ALFCGIVW",
}
_CTD_GROUP2 = {
    "hydrophobicity_PRAM900101": "GASTPHY",
    "hydrophobicity_ARGP820101": "RAHCKMV",
    "hydrophobicity_ZIMJ680101": "HMCKV",
    "hydrophobicity_PONP930101": "GRHA",
    "hydrophobicity_CASG920101": "AHYMLV",
    "hydrophobicity_ENGD860101": "SGTAW",
    "hydrophobicity_FASG890101": "NTPG",
    "normwaalsvolume": "NVEQIL",
    "polarity": "PATGS",
    "polarizability": "CPNVEQIL",
    "charge": "ANCQGHILMFPSTWYV",
    "secondarystruct": "VIYCWFT",
    "solventaccess": "RKQEND",
}
_CTD_GROUP3 = {
    "hydrophobicity_PRAM900101": "CLVIMFW",
    "hydrophobicity_ARGP820101": "LYPFIW",
    "hydrophobicity_ZIMJ680101": "LPFYI",
    "hydrophobicity_PONP930101": "YMFWLCVI",
    "hydrophobicity_CASG920101": "FIWC",
    "hydrophobicity_ENGD860101": "CVLIMF",
    "hydrophobicity_FASG890101": "AYHWVMFLIC",
    "normwaalsvolume": "MHKFRYW",
    "polarity": "HQRKNED",
    "polarizability": "KMHFRYW",
    "charge": "DE",
    "secondarystruct": "GNPSD",
    "solventaccess": "MSPTHY",
}

# Chou PAAC.txt order is ARNDCQEGHILKMFPSTWYV. Values are copied into `AA` order below.
_PAAC_ORDER = "ARNDCQEGHILKMFPSTWYV"
_HYDROPHOBICITY_CHOU = np.array(
    [0.62, -2.53, -0.78, -0.9, 0.29, -0.85, -0.74, 0.48, -0.4, 1.38, 1.06, -1.5, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08],
    dtype=np.float64,
)
_HYDROPHILICITY = np.array(
    [-0.5, 3, 0.2, 3, -1, 0.2, 3, 0, -0.5, -1.8, -1.8, 3, -1.3, -2.5, 0, 0.3, -0.4, -3.4, -2.3, -1.5],
    dtype=np.float64,
)
_SIDE_MASS = np.array(
    [15, 101, 58, 59, 47, 72, 73, 1, 82, 57, 57, 73, 75, 91, 42, 31, 45, 130, 107, 43],
    dtype=np.float64,
)

# Eleven scales, each a length-20 vector in `AA` order (ACDEFGHIKLMNPQRSTVWY).
# Sources: Chou PAAC.txt; Kyte-Doolittle; Grantham 1974; standard pKa / pI / mass tables.
_AAI_NAMES = (
    "chou_hydrophobicity",
    "hopp_woods_hydrophilicity",
    "side_chain_mass",
    "kyte_doolittle",
    "grantham_polarity",
    "grantham_volume",
    "pi",
    "pka_carboxyl",
    "pka_amino",
    "charge_ph7",
    "molecular_weight",
)
_KYTE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
_GRANTHAM_POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3, "G": 9.0,
    "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2,
    "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
_GRANTHAM_VOLUME = {
    "A": 31.0, "R": 124.0, "N": 56.0, "D": 54.0, "C": 55.0, "Q": 85.0, "E": 83.0, "G": 3.0,
    "H": 96.0, "I": 111.0, "L": 111.0, "K": 119.0, "M": 105.0, "F": 132.0, "P": 32.5,
    "S": 32.0, "T": 61.0, "W": 170.0, "Y": 136.0, "V": 84.0,
}
_PI = {
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07, "Q": 5.65, "E": 3.22, "G": 5.97,
    "H": 7.59, "I": 6.02, "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.30, "S": 5.68,
    "T": 5.60, "W": 5.89, "Y": 5.66, "V": 5.96,
}
_PKA1 = {
    "A": 2.34, "R": 2.17, "N": 2.02, "D": 1.88, "C": 1.96, "Q": 2.17, "E": 2.19, "G": 2.34,
    "H": 1.82, "I": 2.36, "L": 2.36, "K": 2.18, "M": 2.28, "F": 1.83, "P": 1.99, "S": 2.21,
    "T": 2.09, "W": 2.38, "Y": 2.20, "V": 2.32,
}
_PKA2 = {
    "A": 9.69, "R": 9.04, "N": 8.80, "D": 9.60, "C": 10.28, "Q": 9.13, "E": 9.67, "G": 9.60,
    "H": 9.17, "I": 9.68, "L": 9.60, "K": 8.95, "M": 9.21, "F": 9.13, "P": 10.96, "S": 9.15,
    "T": 9.10, "W": 9.39, "Y": 9.11, "V": 9.62,
}
_CHARGE = {aa: (1.0 if aa in "KRH" else -1.0 if aa in "DE" else 0.0) for aa in AA}
_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16, "Q": 146.15, "E": 147.13,
    "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19,
    "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

CTD_DIM = 13 * 21
PAAC_DIM = N_AA + 1
APAAC_DIM = N_AA + 2
AAI_DIM = len(_AAI_NAMES)
NTERM_DIM = N_AA * N_AA
FP_DIMS = {"estate": 79, "maccs": 166, "pubchem": 881, "fp4": 307}


def _paac_in_aa_order(values: np.ndarray) -> np.ndarray:
    out = np.empty(N_AA, dtype=np.float64)
    for letter, value in zip(_PAAC_ORDER, values):
        out[AA_INDEX[letter]] = value
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = np.sqrt(np.mean(centered**2))
    if scale < 1e-12:
        return np.zeros_like(values)
    return centered / scale


_H1 = _zscore(_paac_in_aa_order(_HYDROPHOBICITY_CHOU))
_H2 = _zscore(_paac_in_aa_order(_HYDROPHILICITY))
_MASS = _zscore(_paac_in_aa_order(_SIDE_MASS))
_PAAC_PROPS = np.vstack([_H1, _H2, _MASS])


def _dict_in_aa_order(table: dict[str, float]) -> np.ndarray:
    return np.array([table[aa] for aa in AA], dtype=np.float64)


_AAI_TABLE = np.column_stack(
    [
        _paac_in_aa_order(_HYDROPHOBICITY_CHOU),
        _paac_in_aa_order(_HYDROPHILICITY),
        _paac_in_aa_order(_SIDE_MASS),
        _dict_in_aa_order(_KYTE),
        _dict_in_aa_order(_GRANTHAM_POLARITY),
        _dict_in_aa_order(_GRANTHAM_VOLUME),
        _dict_in_aa_order(_PI),
        _dict_in_aa_order(_PKA1),
        _dict_in_aa_order(_PKA2),
        _dict_in_aa_order(_CHARGE),
        _dict_in_aa_order(_MW),
    ]
)


def _group_index(letters: str) -> np.ndarray:
    out = np.zeros(N_AA, dtype=np.int8)
    for ch in letters:
        out[AA_INDEX[ch]] = 1
    return out


_CTD_MASKS = []
for _name in _CTD_PROPERTIES:
    _CTD_MASKS.append(
        (
            _group_index(_CTD_GROUP1[_name]),
            _group_index(_CTD_GROUP2[_name]),
            _group_index(_CTD_GROUP3[_name]),
        )
    )


def ctd_groups_partition_alphabet() -> bool:
    """True when every property's three groups are a partition of the 20 letters."""
    for masks in _CTD_MASKS:
        total = masks[0] + masks[1] + masks[2]
        if not np.array_equal(total, np.ones(N_AA, dtype=np.int8)):
            return False
    return True


def _encode_groups(seq: str) -> list[np.ndarray]:
    idx = np.array([AA_INDEX[ch] for ch in seq], dtype=np.int64)
    encoded = []
    for masks in _CTD_MASKS:
        groups = np.zeros(len(seq), dtype=np.int8)
        groups[masks[0][idx] == 1] = 1
        groups[masks[1][idx] == 1] = 2
        groups[masks[2][idx] == 1] = 3
        encoded.append(groups)
    return encoded


def ctd_features(seqs: list[str]) -> np.ndarray:
    """Return (n, 273) composition, transition and distribution features."""
    out = np.zeros((len(seqs), CTD_DIM), dtype=np.float64)
    for row, seq in enumerate(seqs):
        length = len(seq)
        if length == 0:
            raise ValueError("Empty peptide")
        cursor = 0
        for groups in _encode_groups(seq):
            for label in (1, 2, 3):
                out[row, cursor] = np.mean(groups == label)
                cursor += 1
            if length >= 2:
                left = groups[:-1]
                right = groups[1:]
                for a, b in ((1, 2), (1, 3), (2, 3)):
                    out[row, cursor] = np.mean(
                        ((left == a) & (right == b)) | ((left == b) & (right == a))
                    )
                    cursor += 1
            else:
                cursor += 3
            positions = np.arange(1, length + 1, dtype=np.float64) / length
            for label in (1, 2, 3):
                hits = positions[groups == label]
                if hits.size == 0:
                    cursor += 5
                    continue
                # First, 25%, 50%, 75% and last occurrence. Fractions of length.
                quantiles = (0.0, 0.25, 0.50, 0.75, 1.0)
                for q in quantiles:
                    index = int(np.floor(q * (hits.size - 1)))
                    out[row, cursor] = hits[index]
                    cursor += 1
        if cursor != CTD_DIM:
            raise RuntimeError(f"CTD wrote {cursor} columns, expected {CTD_DIM}")
    return out


def _sequence_correlation(seq: str, scale: np.ndarray, lag: int) -> float:
    if len(seq) <= lag:
        return 0.0
    values = scale[np.array([AA_INDEX[ch] for ch in seq], dtype=np.int64)]
    return float(np.mean(values[:-lag] * values[lag:]))


def _paac_theta(seq: str, lag: int) -> float:
    if len(seq) <= lag:
        return 0.0
    idx = np.array([AA_INDEX[ch] for ch in seq], dtype=np.int64)
    left = _PAAC_PROPS[:, idx[:-lag]]
    right = _PAAC_PROPS[:, idx[lag:]]
    return float(np.mean(np.mean((left - right) ** 2, axis=0)))


def paac_features(seqs: list[str], lam: int = 1, weight: float = 0.05) -> np.ndarray:
    """Pseudo amino acid composition. One lambda gives 21 columns."""
    out = np.zeros((len(seqs), N_AA + lam), dtype=np.float64)
    for row, seq in enumerate(seqs):
        if not seq:
            raise ValueError("Empty peptide")
        freq = np.zeros(N_AA, dtype=np.float64)
        for ch in seq:
            freq[AA_INDEX[ch]] += 1.0
        freq /= len(seq)
        theta = np.array([_paac_theta(seq, lag) for lag in range(1, lam + 1)])
        denom = 1.0 + weight * float(theta.sum())
        out[row, :N_AA] = freq / denom
        out[row, N_AA:] = weight * theta / denom
    return out


def apaac_features(seqs: list[str], lam: int = 1, weight: float = 0.05) -> np.ndarray:
    """Amphiphilic PAAC. One lambda gives 22 columns."""
    out = np.zeros((len(seqs), N_AA + 2 * lam), dtype=np.float64)
    for row, seq in enumerate(seqs):
        if not seq:
            raise ValueError("Empty peptide")
        freq = np.zeros(N_AA, dtype=np.float64)
        for ch in seq:
            freq[AA_INDEX[ch]] += 1.0
        freq /= len(seq)
        theta = []
        for lag in range(1, lam + 1):
            theta.append(_sequence_correlation(seq, _H1, lag))
            theta.append(_sequence_correlation(seq, _H2, lag))
        theta_arr = np.array(theta, dtype=np.float64)
        denom = 1.0 + weight * float(theta_arr.sum())
        out[row, :N_AA] = freq / denom
        out[row, N_AA:] = weight * theta_arr / denom
    return out


def nterm_dipeptide_features(seqs: list[str]) -> np.ndarray:
    """One-hot of the first dipeptide only. Zeros when the peptide has one residue.

    Unlike dipeptide composition, later pairs do not dilute this coordinate.
    """
    out = np.zeros((len(seqs), NTERM_DIM), dtype=np.float64)
    for row, seq in enumerate(seqs):
        if len(seq) < 2:
            continue
        out[row, AA_INDEX[seq[0]] * N_AA + AA_INDEX[seq[1]]] = 1.0
    return out


def aai_features(seqs: list[str]) -> np.ndarray:
    """Mean of 11 physicochemical scales. Shape (n, 11)."""
    out = np.zeros((len(seqs), AAI_DIM), dtype=np.float64)
    for row, seq in enumerate(seqs):
        if not seq:
            raise ValueError("Empty peptide")
        idx = np.array([AA_INDEX[ch] for ch in seq], dtype=np.int64)
        out[row] = _AAI_TABLE[idx].mean(axis=0)
    return out


def _molecules(seqs: list[str]):
    from rdkit import Chem

    mols = []
    for seq in seqs:
        mol = Chem.MolFromSequence(seq)
        if mol is None:
            raise ValueError(f"RDKit could not build a peptide from {seq!r}")
        mols.append(mol)
    return mols


def fingerprint_features(seqs: list[str], kind: str) -> np.ndarray:
    """EState, MACCS, PubChem, or FP4 (Laggner, 307 bits)."""
    from skfp.fingerprints import (
        EStateFingerprint,
        LaggnerFingerprint,
        MACCSFingerprint,
        PubChemFingerprint,
    )

    builders = {
        "estate": EStateFingerprint,
        "maccs": MACCSFingerprint,
        "pubchem": PubChemFingerprint,
        "fp4": LaggnerFingerprint,
    }
    if kind not in builders:
        raise ValueError(f"Unknown fingerprint {kind!r}")
    encoded = np.asarray(builders[kind](n_jobs=1).transform(_molecules(seqs)), dtype=np.float64)
    if encoded.shape != (len(seqs), FP_DIMS[kind]):
        raise RuntimeError(f"{kind} produced shape {encoded.shape}")
    return encoded
