import pandas as pd
from smarts import fg_list_original_dict
import rdkit.Chem as Chem
from collections import Counter

''' helper function that takes a compound's smile string and returns a list of functional groups 
present in the compound using substruct match'''

path = 'CHEM_ML_PREPROC.csv'
data_df = pd.read_csv(path)

def identify_functional_groups(smiles_string):
    molecule = Chem.MolFromSmiles(smiles_string)
    functional_group_matches = []
    for group_name, group_smarts in fg_list_original_dict.items():
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(group_smarts)):
            functional_group_matches.append(group_name)
    if functional_group_matches == []: return ['Other']
    else: return functional_group_matches
    
data_df['fgroup'] = data_df['smiles'].apply(identify_functional_groups)

def find_most_common_groups(n):
    data_fgroups = [item for sublist in data_df['fgroup'] for item in sublist]
    fgroup_counts = Counter(data_fgroups)
    common_fgroups = []
    print(f'\nThe {n} most common functional groups are: \n')
    for i in range(n):
        print(f'{i+1}\'th most common functional group:')
        element, frequency = fgroup_counts.most_common(n)[i]
        print(f'{element}: {frequency}\n')
        common_fgroups.append(element)
    return common_fgroups

def get_single_common_fgroup(fgroups, common_fgroups):
    # common_fgroups = find_most_common_groups(5)
    
    for fg in fgroups:
        if fg in common_fgroups:
            # discarding the compounds which have multiple common functional groups
            for fg2 in fgroups:
                if fg != fg2 and fg2 in common_fgroups:
                    return 'Complex'
            return fg
    
    return 'Other'
        
def get_common_fgroups_df(n):
    common_fgroups = find_most_common_groups(n)
    
    data_df['fg_label'] = data_df['fgroup'].apply(lambda x: get_single_common_fgroup(x, common_fgroups))
    
    return data_df

if __name__ == '__main__':
    ''' code that prints the frequency of fgroups in the dataset '''
    # find_most_common_groups(5)
    new_df = get_common_fgroups_df(5)
    print(new_df)
    
        

    # print(data_df['fgroup'].value_counts())

