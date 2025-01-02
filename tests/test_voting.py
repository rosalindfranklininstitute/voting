import os
import voting


def test_main(tmp_path):
    os.chdir(tmp_path)
    voting.main([os.path.join(os.path.dirname(__file__), "test_votes.csv")])
