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
Note that the cluster 0 consists very small clusters lumbed together.
## Parameters
Each parameter is explained in the -help argument, but some arguments may not be clear, so I will explain them here:

1. <code>-n</code> or <code>--nclusters</code> : Setting the number here to be 200 won't really produce 200 cluster in the output nodes, but it will divide the data 200 times, but some divisions will end up being noisy and contain virtually no data which will be filtered automatically in the code. It's generally good to set the cluster number to be 1/100 of the data size to produce meaningful clusters.
2. <code>-d</code> or <code>--ndivisable</code> : Leaving the default value here which is 100 will probably be good enough but if your data is noisy you can decrease it, but no that when decreasing it, it's advised to increase <code>--nclusters</code> .
3. <code>-p</code> or <code>--percentage_filtered</code> : It's 90 by default, but you can increase it or decrease it if you find the keywords not satisfying.

## Output
After entering the nodes and edges to gephi, if you want the graph to appear as it does in the following screenshot:
![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/19c70146-29bc-48d3-a0b7-526dab9301b5)
<br/>
You can use OpenOrd layout with the parameters shown here:
<br/>
![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/e1000d8f-e4e4-44bb-9b04-05641a79ba92)
<br/>

Then use a preset with these details:
<br/>
![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/71615152-3c55-4947-ae29-27b1d1e93d24)
<br/>


The following images are screenshots from the zoomed in graph which had some notible color configuration where:
1. Cluster 0 (the combined small clusters) is colored in black.
2. Clusters with lower than 2 percent of the data is colored in grey.
You can toggle the label to show the URL instead of the html title if you want.
<br/>

![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/159ac4ed-9436-4b29-b4d5-bfcc49376262)

<br/>

![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/9f630bbd-d822-4013-a998-651f52592575)
<br/>

![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/df5ea26d-886a-4e23-9387-21fc761d8707)
<br/>

![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/96c5d043-293e-45ca-b068-c5d4e69c0247)
<br/>

![image](https://github.com/AmrSheta22/meshwarc/assets/78879883/18001861-8ffb-457c-b70e-53ac2e7d9988)
<br/>

