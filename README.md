# meshwarc
A semantic-based network representation of the web archive.
In this project we process a json file consisting of html data _which can be produced using warchtml here https://github.com/AmrSheta22/warchtml _ to output a graph based on the similarity between each pair of html pages, then use topic modelling to create clusters of pages containing the same topic.
## Usage
You can install the projects requirments using the following code:
```
pip install -r requirments.txt
```
To run the script on an input file you can run something like this:
```
python3 main.py -i ./data/ -o ./data/ -n 200 -d 10 -e ./data/embeddings.txt -p 90 -s 0.37 -t 10
```
The code should output pickled lists which are used to speed up the process in case an error happened in the middle of the code, and three csv files: edges, nodes and merged_tfidf, you can use the edges and nodes as an input to Gephi to show a graph of the data directly.

## Parameters
Each parameter is explained in the -help argument, but some arguments may not be clear, so I will explain them here:
1- <code>-n</code> or <code>--nclusters</code> : Setting the number here to be 200 won't really produce 200 cluster in the output nodes, but it will divide the data 200 times, but some divisions will end up being noisy and contain virtually no data which will be filtered automatically in the code. It's generally good to set the cluster number to be 1/100 of the data size to produce meaningful clusters.
\\
2- <code>-d</code> or <code>--ndivisable</code> : Leaving the default value here which is 100 will probably be good enough but if your data is noisy you can decrease it, but no that when decreasing it, it's advised to increase <code>--nclusters</code> .
\\
3- <code>-p</code> or <code>--percentage_filtered</code> : It's 90 by default, but you can increase it or decrease it if you find the keywords not satisfying.
