import meshwarc_utils as mu
import pandas as pd
import argparse

# parse all the arguments to the main function
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="input json file path",
    required=True,
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="output json file path",
    default="./",
)
parser.add_argument(
    "--nclusters",
    "-n",
    type=int,
    help="number of clusters",
    default=200,
)
parser.add_argument(
    "--ndivisable",
    "-d",
    type=int,
    help="divisable number",
    default=10,
)
parser.add_argument(
    "--embeddings",
    "-e",
    type=str,
    help="embeddings file path",
    default=None,
)
parser.add_argument(
    "--percentage_filtered",
    "-p",
    type=int,
    help="percentage of filtered text from using attention",
    default=90,
)
parser.add_argument(
    "--similarity_threshold",
    "-s",
    type=float,
    help="similarity threshold for the graph edge construction",
    default=0.37,
)
parser.add_argument(
    "--top_n",
    "-t",
    type=int,
    help="top n words to be used by tfidf model",
    default=10,
)


# create main
def main():
    # parse the arguments
    args = parser.parse_args()
    # create the meshwarc object
    similar_df, cluster_df, _, _, merged_tfidf = mu.meshwarc(
        path=args.input,
        top_n=args.top_n,
        similarity_threshold=args.similarity_threshold,
        divisable_cluster_size=args.ndivisable,
        number_of_clusters=args.nclusters,
        minimum_cluster_size=100,
        percentage_filtered=args.percentage_filtered,
        embedding=args.embeddings,
    )
    # save the results
    similar_df.to_csv(args.output + "edges.csv")
    cluster_df.to_csv(args.output + "nodes.csv")
    merged_tfidf.to_csv(args.output + "tfidf_df.csv")


if __name__ == "__main__":
    main()
