import os
import voting


def test_main(tmp_path):
    os.chdir(tmp_path)
    voting.main([os.path.join(os.path.dirname(__file__), "test_votes.csv")])


def test_main_make_plots(tmp_path):
    os.chdir(tmp_path)
    voting.main(
        [os.path.join(os.path.dirname(__file__), "test_votes.csv"), "--make_plots"]
    )
    os.path.exists("election/vote_matrix.png")
    os.path.exists("election/location_vote_matrix.png")
    os.path.exists("election/num_approved.png")
    os.path.exists("election/clustering.png")
