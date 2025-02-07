import unittest
import pandas as pd

def calculate_v_index(df_dependent, df_repo):
    """
    Calculate V-Index of a software package.
    
    V-Index is N where N is the number of first-order dependents that have
    at least N second-order dependents.
    """
    data = {}

    for _, row in df_repo.iterrows():
        repo_url = row['repo_url']
        # convert first order dependents string into a list
        first_order_dependents = row['list_of_dependents_in_oso'].strip("[]").replace("'", "").split()
        
        # filter df_dependent for rows matching first order dependents
        first_order_df = df_dependent[df_dependent['package_artifact_name'].isin(first_order_dependents)].copy()

        # sort in descending order by second-order dependents counts
        first_order_df.sort_values(by='num_dependents', ascending=False, inplace=True)
        
        # convert second order dependents counts into a list
        second_order_counts = first_order_df['num_dependents'].tolist()
        
        # v-index
        # Find the the number of first-order dependents N that have at least N second-order dependents
        v_index = 0
        for i, count in enumerate(second_order_counts):
            # i is zero-based, so the candidate N is (i+1).
            # If count < (i+1), we can't claim an index of (i+1).
            if (i + 1) > count:
                v_index = i  # the largest index we could achieve so far
                break
        else:
            # If we never break, it means *all* dependencies had enough second-order
            # so the V-Index equals the total number of first-order dependencies
            v_index = len(second_order_counts)

        data[repo_url] = v_index

    return data

class TestCalculateVIndex(unittest.TestCase):
    def setUp(self):
        # Create a sample df_dependent DataFrame
        self.df_dependent = pd.DataFrame({
            'package_artifact_name': ['curvefi', 'gnosis', 'apeworx', 'nucypher', 'uniswap'],
            'num_dependents': [3, 3, 2, 8, 7]  # Example second-order dependency counts
        })

        # Create a sample df_repo DataFrame
        self.df_repo = pd.DataFrame({
            'repo_url': ['repo1', 'repo2'],
            'list_of_dependents_in_oso': ["['curvefi' 'gnosis' 'apeworx']", "['nucypher' 'uniswap']"]
        })

    def test_v_index(self):
        expected_output = {
            'repo1': 2,  # 1 first-order dependents have at least 2 second-order dependents
            'repo2': 2   # 2 first-order dependents have at least 2 second-order dependents
        }

        result = calculate_v_index(self.df_dependent, self.df_repo)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()