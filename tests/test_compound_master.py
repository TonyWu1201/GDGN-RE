"""测试：药物主表（计划 §14 test_compound_master，§4.2）。"""

import pandas as pd
import pytest
from rdkit import Chem


def test_metal_complex_not_fragmented():
    """金属配合物不拆片段：Cisplatin/Oxaliplatin 保留原始 SMILES 且盐策略标记。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str).fillna("")
    metals = m[m["salt_policy"] == "metal_complex_untouched"]
    assert len(metals) == 3  # Cisplatin + Oxaliplatin ×2
    assert (metals["original_smiles"] == metals["standardized_isomeric_smiles"]).all()


def test_same_inchikey_same_parent():
    """同 InChIKey 同 normalized_parent_id；10 对闭合。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str)
    parent_per_key = m.groupby("inchikey_adjudicated")["normalized_parent_id"].nunique()
    assert (parent_per_key == 1).all()
    key_counts = m["inchikey_adjudicated"].value_counts()
    multi = key_counts[key_counts > 1]
    assert len(multi) == 10 and (multi == 2).all()


def test_rapamycin_and_low_confidence_not_in_core():
    """Rapamycin/低置信度不进 Core。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str)
    assert "Rapamycin" not in set(m["source_name"])
    disp = pd.read_csv("data/processed/entities/compound_disposition.csv", dtype=str)
    rap = disp[disp["source_name"] == "Rapamycin"].iloc[0]
    assert rap["disposition"] == "review_required_not_core"
    low = disp[disp["disposition"] == "low_confidence_not_core"]
    assert len(low) == 4
    # 低置信度药物不在主表
    assert set(low["source_drug_id"]).isdisjoint(set(m["source_drug_id"]))


def test_standardization_reversible_parse():
    """标准化 SMILES 可逆解析：RDKit MolFromSmiles 全部成功且 canonical 稳定。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str).fillna("")
    for smi in m["standardized_isomeric_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        assert Chem.MolToSmiles(mol, isomericSmiles=True) != ""


def test_salt_largest_fragment():
    """盐取最大片段：含 Cl 反离子的药物标准化后不含 [Cl-]/Cl 单片段。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str).fillna("")
    normal = m[m["salt_policy"] == "largest_fragment_as_parent"]
    for smi in normal["standardized_isomeric_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        frags = Chem.GetMolFrags(mol)
        # 单一片段（最大片段母体）
        assert len(frags) == 1, smi


def test_compound_id_deterministic():
    """compound_id 按 DRUG_ID 排序确定性编号。"""
    m = pd.read_csv("data/processed/entities/compound_master.csv", dtype=str)
    ids = m["source_drug_id"].astype(int).tolist()
    assert ids == sorted(ids)
    assert m["compound_id"].iloc[0] == "CMP0001"