"""A declared chiral Morgan fingerprint must separate enantiomers."""

from program.features.build_fingerprints import morgan_generator, smiles_to_fp


def test_enantiomers_have_distinct_fingerprints():
    generator = morgan_generator()
    left = smiles_to_fp("C[C@H](O)F", generator)
    right = smiles_to_fp("C[C@@H](O)F", generator)
    assert left != right
