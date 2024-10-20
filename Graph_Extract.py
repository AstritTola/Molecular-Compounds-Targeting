
import pandas as pd
import numpy as np
import scipy
import networkx as nx
from rdkit import Chem
import dgl

allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
        'possible_chirality_list': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER',
            'misc'
        ],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization_list': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
        'possible_is_aromatic_list': [False, True],
        'possible_is_in_ring_list': [False, True],
        'possible_bond_type_list': [
            'SINGLE',
            'DOUBLE',
            'TRIPLE',
            'AROMATIC',
            'misc',
        ],
        'possible_bond_stereo_list': [
            'STEREONONE',
            'STEREOZ',
            'STEREOE',
            'STEREOCIS',
            'STEREOTRANS',
            'STEREOANY',
        ],
        'possible_is_conjugated_list': [False, True],
    }

def safe_index(l, e):
        """
        Return index of element e in list l. If e is not present, return the last index
        """
        try:
            return l.index(e)
        except:
            return len(l) - 1

# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
        """
        Converts rdkit atom object to feature list of indices
        :param mol: rdkit atom object
        :return: list
        """
        atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
        ]
        return atom_feature

# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]

def get_atom_feature_dims():
        return list(map(len, [
            allowable_features['possible_atomic_num_list'],
            allowable_features['possible_chirality_list'],
            allowable_features['possible_degree_list'],
            allowable_features['possible_formal_charge_list'],
            allowable_features['possible_numH_list'],
            allowable_features['possible_number_radical_e_list'],
            allowable_features['possible_hybridization_list'],
            allowable_features['possible_is_aromatic_list'],
            allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
            safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
            allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
            allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        ]
        return bond_feature

    # uses same molecule as atom_to_feature_vector test
    # bond = mol.GetBondWithIdx(2)  # double bond with stereochem
    # bond_feature = bond_to_feature_vector(bond)
    # assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
        return list(map(len, [
            allowable_features['possible_bond_type_list'],
            allowable_features['possible_bond_stereo_list'],
            allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
        [atomic_num_idx,
         chirality_idx,
         degree_idx,
         formal_charge_idx,
         num_h_idx,
         number_radical_e_idx,
         hybridization_idx,
         is_aromatic_idx,
         is_in_ring_idx] = atom_feature

        feature_dict = {
            'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
            'chirality': allowable_features['possible_chirality_list'][chirality_idx],
            'degree': allowable_features['possible_degree_list'][degree_idx],
            'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
            'num_h': allowable_features['possible_numH_list'][num_h_idx],
            'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
            'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
            'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
            'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
        }

        return feature_dict

# # uses same atom_feature as atom_to_feature_vector test
# atom_feature_dict = atom_feature_vector_to_dict(atom_feature)
# assert atom_feature_dict['atomic_num'] == 6
# assert atom_feature_dict['chirality'] == 'CHI_TETRAHEDRAL_CCW'
# assert atom_feature_dict['degree'] == 4
# assert atom_feature_dict['formal_charge'] == 0
# assert atom_feature_dict['num_h'] == 1
# assert atom_feature_dict['num_rad_e'] == 0
# assert atom_feature_dict['hybridization'] == 'SP3'
# assert atom_feature_dict['is_aromatic'] == False
# assert atom_feature_dict['is_in_ring'] == False

def bond_feature_vector_to_dict(bond_feature):
        [bond_type_idx,
         bond_stereo_idx,
         is_conjugated_idx] = bond_feature

        feature_dict = {
            'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
            'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
            'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
        }

        return feature_dict

def generate_hydrogen_node_id(carbon_node_id, hydrogen_index, M):
        return M + hydrogen_index

# Helper function to connect hydrogen atoms to their corresponding carbon nodes with edges
def connect_hydrogen_atoms(node_features, edge_features, M):
        new_node_features = node_features.copy()
        new_edge_features = edge_features.copy()
        hydrogen_index = 0

        for node_id, features in node_features.items():
            num_hydrogen = features['num_h']
            if num_hydrogen > 0:
                carbon_node_id = node_id
                hydrogen_atomic_num = 1

                for _ in range(num_hydrogen):
                    hydrogen_node_id = generate_hydrogen_node_id(carbon_node_id, hydrogen_index, M)
                    new_node_features[hydrogen_node_id] = {'atomic_num': hydrogen_atomic_num, 'num_h': 0}
                    new_edge_features[(carbon_node_id, hydrogen_node_id)] = {'bond_type': 1}
                    hydrogen_index += 1

        return new_node_features, new_edge_features






def get_info(smiles):

    mol = Chem.MolFromSmiles(smiles)

    node_features = {}
    edge_features = {}

    # Create dictionaries to store node features and edge features
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_features = {'atomic_num': atom.GetAtomicNum(), 'num_h': atom.GetTotalNumHs()}
        node_features[atom_idx] = atom_features

    #print(node_features)

    # Populate edge features dictionary
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_feature = bond_to_feature_vector(bond)
        bond_features = {'bond_type': bond_feature[0] + 1}
        edge_features[(begin_atom_idx, end_atom_idx)] = bond_features

    M = len(node_features)

    # Connect hydrogen atoms to their corresponding carbon nodes
    new_node_features, new_edge_features = connect_hydrogen_atoms(node_features, edge_features, M)

    # Print the updated node features
    #print("Updated Node Features:")
    #for node_id, features in new_node_features.items():
        #print(f"Node {node_id}: {features}")

    # Print the updated edge features
    #print("\nUpdated Edge Features:")
    #for edge, features in new_edge_features.items():
        #print(f"Edge {edge}: {features}")
    print(new_node_features, new_edge_features)

    return new_node_features, new_edge_features
