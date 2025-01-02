#
# Copyright (C) 2025 Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPL license.
#

import itertools
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
from argparse import ArgumentParser
from matplotlib import pylab
from scipy.cluster.hierarchy import dendrogram, linkage


def make_vote_matrix(voters: list, num_candidates: int) -> np.ndarray:
    """
    Construct the vote matrix indicating which voters approve of which
    candidate where each row corresponds to a voter and each column
    corresponds to a candidate

    Args:
        voters: A list of which candidate each voter voted for
        num_candidates: The number of candidates

    Returns:
        The vote matrix

    """
    voting_matrix = np.zeros((len(voters), num_candidates), dtype=bool)
    for i, voter in enumerate(voters):
        for v in voter:
            voting_matrix[i, v] = True
    return voting_matrix


def read_voting_data(filename: str) -> tuple:
    """
    Read the vote data from the CSV file

    Args:
        filename: The CSV filename

    Returns:
        A tuple with (candidates, vote_matrix)

    """

    # Parse the names of the chosen candidates. The entry is given by a string
    # delimited by a ";" where each entry contains the following, where we just
    # want the name: ${NAME} (${THEME}) - "Description";
    def parse_choices(choices):
        choices_list = []
        for v in choices.split(";"):
            v = v.strip()
            if v != "":
                choices_list.append(v.split("(")[0].strip())
        return choices_list

    # Read the CSV file
    voting_data = pd.read_csv(filename)

    # Check the columns are as we expect
    assert voting_data.shape[1] == 9
    assert voting_data.keys()[0] == "ID"
    assert voting_data.keys()[1] == "Start time"
    assert voting_data.keys()[2] == "Completion time"
    assert voting_data.keys()[3] == "Email"
    assert voting_data.keys()[4] == "Name"
    assert voting_data.keys()[5] == "Last modified time"
    assert voting_data.keys()[6] == "To which theme do you belong at the RFI?"
    assert (
        voting_data.keys()[7]
        == "Who do you want to vote for?  Each postdoc can vote for up to 7 candidates. "
    )
    assert (
        voting_data.keys()[8] == "Do you have any suggestions to the future committee? "
    )

    # This is the column to read the votes from
    location_column, choices_column = 6, 7

    # For each voter get the choices as a list
    choices = [parse_choices(c) for c in voting_data.iloc[:, choices_column]]

    # Get the location of each voter
    vote_location = voting_data.iloc[:, location_column]

    # Get the set of candidates and assign a numerical value to each candidate
    candidates = np.array(sorted(list(set([cc for c in choices for cc in c]))))
    lookup = dict(zip(candidates, range(len(candidates))))
    choices = [[lookup[rr] for rr in r] for r in choices]

    # Return the candidates and the vote matrix
    return candidates, make_vote_matrix(choices, len(candidates)), vote_location


def count_votes(vote_matrix: np.ndarray, num_winners: int):
    """
    Count the votes to determine who is elected

    Args:
        vote_matrix: The matrix of who votes for who
        num_winners: The number of winning candidates

    """

    # The vote matrix has dimension (num_voters, num_candidates)
    num_voters, num_candidates = vote_matrix.shape

    # Check we have enough candidates
    assert num_candidates >= num_winners

    # Create a vector giving whether a candidate is elected, tied, or nothing
    outcome = np.full(num_candidates, "", dtype=object)

    # Loop until we have filled each position
    while np.sum(outcome == "Elected") < num_winners and np.sum(outcome == "Tied") == 0:

        # Compute the number of winning candidates approved by each voter
        num_approved = np.sum((outcome == "Elected")[None, :] * vote_matrix, axis=1)
        assert len(num_approved) == num_voters
        assert np.all(num_approved >= 0)
        assert np.all(num_approved <= num_winners)

        # Compute the weight given to each voter's ballot
        weight = 1 / (num_approved + 1)
        assert np.all(weight >= 1 / (num_winners + 1))
        assert np.all(weight <= 1)

        # Select only those candidates who are not in the winning set
        select = np.where(outcome != "Elected")[0]

        # Count the weighted and unweighted sum of votes for each candidate
        # whilst ignoring all candidates who are already winners
        count = (
            np.stack(
                [
                    np.sum(weight[:, None] * vote_matrix, axis=0),
                    np.sum(vote_matrix, axis=0),
                ]
            ).T
            * (outcome != "Elected")[:, None]
        )

        # Check vote count and ensure that reweighted vote is less than total vote
        assert count.shape == (num_candidates, 2)
        assert np.all(count >= 0)
        assert np.all(count <= num_voters)
        assert np.all(count[:, 0] <= count[:, 1])

        # First find all candidates with the maximum weighted vote count. Then
        # of those candidates find those with the maximum unweighted vote
        # count. These will be our winners.
        index = np.where(count[:, 0] == np.max(count[:, 0]))[0]
        index = index[np.where(count[index, 1] == np.max(count[index, 1]))[0]]
        assert np.all(count[index, 0] == np.max(count[:, 0]))
        assert np.all(count[index, 1] == np.max(count[index, 1]))

        # If there are enough spaces left then add all winning candidates
        # according to the weighted sum.
        if len(index) <= num_winners - np.sum(outcome == "Elected"):
            outcome[index] = "Elected"
        else:
            outcome[index] = "Tied"
        assert np.all((outcome == "") | (outcome == "Elected") | (outcome == "Tied"))

        # If there are any number of tied candidates, the maximum number of
        # winners should be less than num_winners. Otherwise, it should be
        # less than or equal to num_winners.
        assert np.sum(outcome == "Elected") <= num_winners - (
            np.sum(outcome == "Tied") > 0
        )

        # Yield the results
        yield {
            "num_approved": num_approved,
            "weight": weight,
            "count": count,
            "index": index,
            "select": select,
            "outcome": outcome.copy(),
        }


def print_results(results: list, candidates: list):
    """
    Print the voting results

    Args:
        results: The result of each voting round
        candidates: The list of candidates

    """

    def get_result_rows(candidates, result):

        # Select only those active candidates
        select = result["select"]

        # Create the table rows
        return [
            (name, score, count, outcome)
            for name, score, count, outcome in zip(
                candidates[select],
                result["count"][select, 0],
                result["count"][select, 1],
                result["outcome"][select],
            )
        ]

    # Print the candidate names
    print("💀 Candidates 💀")
    print("")
    print(tabulate([[c] for c in sorted(candidates)], headers=["Names"]))
    print("")

    # Scientific progress
    progress = "🥼🦠🧬🧫🧪🔬🔥"

    # Iterate through the voting rounds and process the results
    final_results_table = []
    for i, result in enumerate(results):

        # Get an emoji
        emoji = progress[i % len(progress)]

        # Create the outcome array
        final_results_table.extend(
            list(zip(candidates[result["index"]], result["outcome"][result["index"]]))
        )

        # Sort the results into rows of (name, score, count)
        voting_round_table = reversed(
            sorted(
                get_result_rows(candidates, result),
                key=lambda x: (x[1], x[2]),
            )
        )

        # Print the voting round table
        print("%s Voting round %d %s" % (emoji, i + 1, emoji))
        print("")
        print(
            tabulate(voting_round_table, headers=["Name", "Score", "Total", "Outcome"])
        )
        print("")

    # Print the final results
    print("💥 Final results 💥")
    print("")
    print(tabulate(final_results_table, headers=["Name", "Outcome", ""]))
    print("")
    print("🎉 " * len(candidates))


def analyse_results(
    candidates: list, vote_matrix: np.ndarray, vote_location: list, results: list
):
    """
    Analyse the election results

    Args:
        candidates: The candidates
        vote_matrix: The vote matrix
        vote_location: The voter location
        results: The voting results

    """

    def plot_vote_matrix(candidates, vote_matrix, outcome, num_approved):
        mask = vote_matrix & (outcome == "Elected")[None, :]
        vote_matrix = vote_matrix.copy().astype(int)
        vote_matrix[mask] = 2
        fig, ax = pylab.subplots(constrained_layout=True)
        ax.imshow(vote_matrix.T)
        ax.set_title("Matrix of candidate approval")
        ax.set_xlabel("Number of elected approved by each voter")
        ax.set_yticks(range(len(candidates)), candidates)
        ax.set_xticks(range(vote_matrix.shape[0]), num_approved)
        fig.savefig("election/vote_matrix.png", dpi=600, bbox_inches="tight")
        pylab.close()

    def plot_location_vote_matrix(candidates, vote_matrix, vote_location):
        vote_location = [str(l).split("-")[0].strip() for l in vote_location]
        locations = sorted(list(set(vote_location)))
        lookup = dict((l, i) for i, l in enumerate(locations))
        location_vote_matrix = np.zeros((len(candidates), len(locations)))
        for i in range(vote_matrix.shape[0]):
            location_vote_matrix[:, lookup[vote_location[i]]] += vote_matrix[i, :]
        fig, ax = pylab.subplots(constrained_layout=True)
        ax.imshow(location_vote_matrix)
        ax.set_title("Matrix of candidates approval by location")
        ax.set_yticks(range(len(candidates)), candidates)
        ax.set_xticks(range(len(locations)), locations)
        ax.tick_params(axis="x", labelrotation=90)
        fig.savefig("election/location_vote_matrix.png", dpi=600, bbox_inches="tight")
        pylab.close()

    def plot_num_approved(vote_matrix, num_approved):
        fig, ax = pylab.subplots(constrained_layout=True)
        ax.set_title("Histogram of number of candidates approved")
        ax.hist(num_approved, bins=np.arange(max(num_approved) + 2) + 0.5)
        ax.set_xlabel("Num approved")
        fig.savefig("election/num_approved.png", dpi=600, bbox_inches="tight")
        pylab.close()

    def plot_clustering(vote_matrix):
        Z = linkage(vote_matrix.T, "ward")
        fig, ax = pylab.subplots(constrained_layout=True)
        dendrogram(Z, ax=ax, labels=candidates)
        ax.set_title("Hierarchical clustering of voting results")
        ax.set_ylabel("Distance")
        ax.tick_params(axis="x", labelrotation=90)
        fig.savefig("election/clustering.png", dpi=600, bbox_inches="tight")
        pylab.close()

    # Get the final outcome and number of approved candidates
    outcome = results[-1]["outcome"]
    num_approved = np.sum((outcome == "Elected")[None, :] * vote_matrix, axis=1)

    # Make the output directory
    if not os.path.exists("election"):
        os.makedirs("election")

    # Generate some plots
    plot_vote_matrix(candidates, vote_matrix, outcome, num_approved)
    plot_location_vote_matrix(candidates, vote_matrix, vote_location)
    plot_num_approved(vote_matrix, num_approved)
    plot_clustering(vote_matrix)


def process_voting_data(voting_data: str, num_winners: int = 7):
    """
    Process the voting data

    Args:
        voting_data: The filename of the voting data file
        num_winners: The number of winning candidates


    """

    # Read the voting data from the CSV file
    candidates, vote_matrix, vote_location = read_voting_data(voting_data)

    # Count the votes and select the winners and the tied votes
    results = list(count_votes(vote_matrix, num_winners))

    # Print the results
    print_results(results, candidates)

    # Do some analysis
    analyse_results(candidates, vote_matrix, vote_location, results)


def main(args=None):
    """
    Main entry point for the script

    """

    # Create the argument parser
    parser = ArgumentParser(description="Count votes for the RFI Post Doc Association!")

    # Add some command line arguments
    parser.add_argument(
        type=str,
        default=None,
        dest="voting_data",
        nargs=1,
        help=(
            """
            The filename for the voting data
            """
        ),
    )
    parser.add_argument(
        "-n",
        "--num_winners",
        type=int,
        default=7,
        dest="num_winners",
        help=(
            """
            The number of elected positions
            """
        ),
    )

    # Parse the command line arguments
    args = parser.parse_args(args=args)

    # Process the voting data
    process_voting_data(args.voting_data[0], args.num_winners)


if __name__ == "__main__":
    main()
