# Tripadvisor analysis and prediction using Graph Neural Networks

Source codes and scraped data used in my diploma thesis,
[Graph neural networks and their application to social network analysis](https://github.com/elkablo/gnn-social-tripadvisor/releases/download/v1.0/thesis.pdf).

## Dependency installation instructions

The following dependency installation instructions are applicable to Ubuntu and Ubuntu derived Linux distributions.

```
# first install git and pip
sudo apt-get install python3-pip git

# afterwards run these command to install Torch[cpu] and related dependencies
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip3 install networkx[default]
pip3 install umap-learn

# finally install pytorch_geometric_temporal with a needed fix from my repository
git clone https://blackhole.sk/~kabel/git/pytorch_geometric_temporal
cd pytorch_geometric_temporal
python3 setup.py build
python3 setup.py install --user
```

## Tripadvisor scraped reviews

Here are GZIP compressed files containing the scraped Tripadvisor reviews used in the thesis.
These contain 3,125,631 reviews of 3,260 U.S. hotels from 2,296,247 authors:

* [`authors.json.gz`](https://github.com/elkablo/gnn-social-tripadvisor/releases/download/v1.0/authors.json.gz)
* [`hotels.json.gz`](https://github.com/elkablo/gnn-social-tripadvisor/releases/download/v1.0/hotels.json.gz)
* [`reviews.json.gz`](https://github.com/elkablo/gnn-social-tripadvisor/releases/download/v1.0/reviews.json.gz)

```
# after downloading, run the following command to compute the checksums
sha256sum authors.json.gz hotels.json.gz reviews.json.gz

# it should output
da741b8b3433aa80fc135912b5d7cdce4b79dfa18b04a7b1e32c56e6c3368c15  authors.json.gz
b205357a12ed61891ee53df96f3ff822087a04aeeadcb0df0fb3c35da1c464e7  hotels.json.gz
a09191f41184625dacdd44a901fd8e91894fee880c31c95f00a51f9dced7b495  reviews.json.gz

# to decompress, run
gunzip authors.json.gz
gunzip hotels.json.gz
gunzip reviews.json.gz
```
