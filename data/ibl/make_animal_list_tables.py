"""

Identify the distinctive animals within the IBL dataset

"""

# Identify unique animals in IBL dataset that enter biased blocks, and save a dictionary with each animal and
# a list of their eids in the biased blocks
import numpy as np
import numpy.random as npr
import json
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from data_utils import mouse_identifier

npr.seed(65)


if __name__ == '__main__':
    ii = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    mice_names = []  # to include animals names

    # Map mouse (subject) name -> parquet file path(s)
    mice_pqt_files = defaultdict(list)

    mice_session_eids = defaultdict(list)  # eids based on the subjects (mice IDs)
    path = Path(__file__).resolve().parent / 'tables_new'

    for dirname, dirs, files in os.walk(path):
        for filename in files:
            filename_without_extension, extension = os.path.splitext(filename)
            if extension == '.pqt':  # each filename is an animal
                pqt_file = os.path.join(dirname + '/' + filename)
                df_trials = pd.read_parquet(pqt_file)
                eids = list(df_trials['session'])
                unique_eids = list(df_trials['session'].unique())
                num_1 += 1

                mouse_name_from_path = mouse_identifier(pqt_file)
                if pqt_file not in mice_pqt_files[mouse_name_from_path]:
                    mice_pqt_files[mouse_name_from_path].append(pqt_file)

                for i, eid_ONE in enumerate(unique_eids):
                    session_trials = df_trials[df_trials['session'] == unique_eids[i]]
                    # below code is because sometimes there is not any probabilityLeft
                    try:
                        probability_stim = session_trials['probabilityLeft']._values
                    except Exception:
                        probability_stim = []
                        ii += 1
                        continue

                    unique_probs = np.unique(probability_stim)
                    # Drop NaNs if present
                    unique_probs = unique_probs[~np.isnan(unique_probs)] if hasattr(unique_probs, "dtype") else unique_probs
                    assess_values = (
                        len(unique_probs) == 3
                        and np.allclose(np.sort(unique_probs.astype(float)), np.array([0.2, 0.5, 0.8]))
                    )

                    if assess_values:
                        num_3 += 1
                        mouse_name = mouse_name_from_path
                        if mouse_name not in mice_names:
                            num_2 += 1
                            mice_names.append(mouse_name)
                        mice_session_eids[mouse_name].append(eid_ONE)

    mice_session_eids_json = json.dumps(mice_session_eids)
    f = open(os.getcwd() + "/mice_session_eids.json", "w")
    f.write(mice_session_eids_json)
    f.close()

    # Also save mapping of mice -> parquet file(s) for downstream scripts
    mice_pqt_files_json = json.dumps(mice_pqt_files)
    f = open(os.getcwd() + "/mice_pqt_files.json", "w")
    f.write(mice_pqt_files_json)
    f.close()

    np.savez(os.getcwd() + '/mice_names.npz', mice_names)
    print('number of animals=', np.array(mice_names).shape)
