import pytest

@pytest.mark.parametrize(
    "input_smiles, expected_output",
    [
        ("CC", "[C:1]([H:3])([H:4])([H:5])[C:7]([H:6])([H:2])([H:8]"),
        ("", "[O:1]=[C:2]([c:3]1[c:4]([H:32])[c:5]2[c:6]([H:33])[c:7](-[c:8]3[n:9][n:10][c:11]([C:12]4([H:34])[C:13]([H:35])([H:36])[C:14]4([H:37])[H:38])[s:15]3)[c:16]([H:39])[n:17][n:18]2[c:19]1[H:40])[N:20]1[C:21]([H:41])([H:42])[C:22]([H:43])([H:44])[N:23]([c:24]2[c:25]([H:45])[c:26]([H:46])[n:27][c:28]([H:47])[c:29]2[H:48])[C:30]([H:49])([H:50])[C:31]1([H:51])[H:52]")
        ("OC(=O)C1=C[NH++]([O-])[CH-]C=C1", None), # Handle this warning: Explicit valence for atom # 5 N, 4, is greater than permitted
        ("[Na+].C[As](O)([O-])=O", None), # Filter out the salts
        ("[H+].[Cl-].CNC1(CCCCC1=O)c2ccccc2Cl", None) # Filter out salts + address 'WARNING: not removing hydrogen atom without neighbors'
    ],
)

def test_preprocess_smiles_basic(input_smiles, expected_output):
    assert preprocess_smiles(input_smiles) == expected_output