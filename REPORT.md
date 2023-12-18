# Graph Neural Networks - Comprehensive Report

## Introduction
In this project, I will first try to understand the theory behind Graphs and their use with Neural Networks. Then, in a second part, I will detail the approach used in order to make Graph Level predictions applied to Molecules Data. Finally, in a third part, I will introduce the work that has been done to implement a Graph Transformer from scratch (*compléter*)

## I - Graph Neural Network Theory

### 1 - What is a Graph ?

A **Graph** can be defined as a Data Structure where elements have nodes, and edges linking nodes between them. 

→ For example, when speaking about molecules, that can be represented with a Graph Structure, the nodes are the atoms of the modecule, and the edges are the chemical bonds linking atoms together.

Graphs are particularly useful to deal with complex interactions between particular entities, and they are widely used with social relations, text and image analysis for example.

Here are a few examples of Graphs that could be used to solve Data Science problematics.

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Sucrose_molecule.svg/1200px-Sucrose_molecule.svg.png" alt="Image 1" width="200">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2352711019302626-gr4.jpg" alt="Image 2" width="200">
</div>

<div align="center">
  <img src="https://deeplobe.ai/wp-content/uploads/2023/06/Object-detection-Real-world-applications-and-benefits.png" alt="Image 1" width="200">
  <img src="https://gatton.uky.edu/sites/default/files/iStock-networkWEB.png" alt="Image 2" width="200">
</div>


From each of these images, we can deduce a Graph Structure made of Nodes and Edges to link them. For the bottom left image, even if it is not straight forward, we could link the two people by a link of type *Human*, and link the two cars with a *vehicle* link for example.


### 2 - What kind of Predictions ?

Graph Data can be used for 3 particular types of prediction tasks : 

#### *2a - Node Level Predictions* : 

It is possible to predict a missing node label from its relations with other nodes. As a straight forward example, a Carbon Dioxyde Molecule has the following Graph Structure : 

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/a/a0/Carbon_dioxide_3D_ball.png" alt="Image Alt Text" width="300"/>
</p>



Now imagine that the right hand side atom of this molecule was unknown. Then, by basic Chemistry knowledge, we can say that this atom is an Oxygen atom, as it has a double chemical bond with the Carbon atom next to it. Learning relations between nodes, the type and attributes of edges that connect them can thus help predicting a Node label.

#### *2b - Link Predictions*

Another possible task to perform is to predict if there is a link between two nodes, and also possibly predict the type of link that links them.

This type of task is very useful for social interaction analysis. Imagine we are considering social relationships between a group of people, but we do not know if two people are actually linked by, let us say, a *friendship* edge.

<p align="center">
<img src="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQDpsitX6WslVX-xPlkvnOX5Se43i9-hJqA63NXyMfZSRyzrdup" alt="Image Alt Text" width="300"/>
</p>

Then, by analysing the graph as a whole, we can make predictions about whether or not these two people are linked. Indeed, if we see that other people are linked to both of them as friends, we could think that the two people have good chances of being friends as well. 

This type of prediction can have amazing value nowadays with the role that play social media and virtual social interactions in our society.

#### *2c - Graph Level Predictions*

This type of predictions is when you wan to predict a characteristic, or a number for example, by looking at a whole known graph, its interactions and properties. For example, when dealing with Recommendation System, you can find yourslef with a graph like this : 

<p align="center">
<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*MVzPPB2RSFNvMsQbzzl0OA.png" alt="Image Alt Text" width="450"/>
</p>

From this clients file represented as a Graph, you can for example decide to send a promotionnal e-mail to these clients based on their preferences.

As such, Graphs can be a valuable Data Structure for many domains, and the potential of predictions has been increasing sharply with the development of Deep Learning algorithms.

In this project, we will focus on a Graph Level Prediction task.

### 3 - Neural Network Theory

#### *3a - Data Specificities*

When using Graph Data, it is important to understand the particular nature of this type of data. Besides the basic structure of a graph (nodes, edges, ...) that we discussed above, here are the specificities that make them different from classical data structures.

- First, every observation in a Graph Dataset can be of different shape and size.
Indeed, molecules for example do not contain the same number of nodes and edges. It does not prevent to analyze and predict the same feature for those graphs. 

- A Graph is also hard to vizualize, as it does not exists in a classical euclidian 2D or 3D space. However, some Python libraries allow to get 2D or 3D representations through dimensionality reduction for example. 

- A Graph also have an infinite possibility of visual representations. Indeed, two different visual representations of a graph containing identical nodes, edges and relations, but differently displayed, refer to a unique Graph observation. We will see an example of this later.

- Increasing the dimension, size and complexity of a Graph can make the analysis very hard to interpret for a human eye.

- A major concept in graph analysis is the Neigborhood, as every node are influenced by their neighborhood nodes and the relation between them. Consider the local structures alongside the global structure is key to obtain a good analysis. The information on the neigborhood of each node can be obtained by transforming the input graph from its general form to a form where each node has a value from a function that takes as input the node itself and its neighborhood. Here is an example, where X<sub>Nb</sub> is the set of nodes connected to the node X<sub>b</sub> : 

<p align="center">
<img src="images/report_imgs/neigb.png" alt="Image Alt Text" width="350"/>
</p>


- Then, when considering a set of nodes (X<sub>1</sub> , ..., X<sub>n</sub>), we induce a node ordering, that we do not wish to impact the neural network prediction. That is, we would like the output to be equivalent for every permutation between nodes. This is called permutation invariance, which is very important to have for graph level predictions which will be our task. We verify this property by finding a prediction function that yields, for every existing perumtation p :
<div align="center">f(p(X)) = f(X)</div>



- In theory, when we consider the edges between the nodes of a graph, we could actually represent our graph in a matrix form, called the adjacency matrix. This type of matrix is a (n x n) matrix, with n being the number of nodes of the graph, and each element of the matrix is 0 or 1 depending on two nodes interacting between each other or not. Here is an example of an adjacency matrix representation, with an H<sub>2</sub>O molecule :

<p align="center">
<img src="https://study.com/cimages/multimages/16/water-2876275_6407686174879164832368.png" alt="Image Alt Text" width="150"/>
</p>


$$
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0 \\
\end{bmatrix}
$$

However, with the classical types of layers used in Graph Neural Networks that we will describe later, the layers do not take the actual adjacency matrix as input 

In summary, a Graph is a singular type of data that needs to be well understood to best adapt the analysis to the taks that is to be performed. Let us know see what are the types of GNN layers, and how they work.

#### *3b - Different types of GNN layers*

When creating a Graph Neural Network, one must choose the type of layers to use to realize the desired task. In general, classical types of layers are Graph Convolutionnal layers, Graph Attentional layers, and Graph Message Passing layers. They differ in the way they diffuse the neigborhood message, so the way that we go from image 1 to image 2 on the above explanation of Neighborhood. These layers are thus the functions taking as parameters the node itself and its neigborhood and send the value h<sub>i</sub> that we see on the image. The way they differ is the weights given to the neighborhood information.

- First, Graph Convolutionnal layers take fixed weights determined by the neigborhood nodes and edges. Here is an illustration, where each node i sends a fixed weight c<sub>bi</sub> :

<p align="center">
<img src="images/report_imgs/gcnconv.png" alt="Image Alt Text" width="300"/>
</p>

Here, the value given to the the node by the layer is given by the following function : 

$$
h_i = φ(x_i, v_j(c_{ji} * ψ(x_j)))
$$

The φ and ψ operators being activation functions, such as sigmoid or ReLU. The v operator is permutation-invariant aggregator, allowing to sum up the neigboorhood information (from every neighborhood node) in one number.

- Then, Graph Attentionnal layers take as weights from the attention given to each part of the neighborhood. Attention in deep learning refers to the process of giving higher importance to some parts of the input compared to other parts. In fact, the weights applied to the nodes are computed from the relation between the 'central' node and its neighbor, so that the kind of relation actually matters. Here is an illustration to sum it up : 

<p align="center">
<img src="images/report_imgs/attention.png" alt="Image Alt Text" width="300"/>
</p>

So that the value given to each node differs only with the weights used, that are functions of two nodes now :

$$
h_i = φ(x_i, v_j(α(x_i, x_j) * ψ(x_j)))
$$

- Another type of layer is Message Passing layer, where 'message' vectors replace the multiplication of weights and an activated value of the neighborhood node, like the following : 

<p align="center">
<img src="images/report_imgs/mp.png" alt="Image Alt Text" width="300"/>
</p>

The value set at each node is 

$$
h_i = φ(x_i, v_j(ψ(X_i, X_j)))
$$

Here, we have m<sub>ji</sub> = ψ(X<sub>i</sub>, X<sub>j</sub>). The difference with attention layers is that the values m<sub>ji</sub> only are used to compute h<sub>i</sub>, and they are not just weights like with other layers.

For this project, during the first phase of Graph Neural Network implementation, we will use Graph Attention layers in the architecture of the model. 

#### *3c - Pooling layers*

Finally, an inportant part of Graph Neural Network is Graph Pooling, especially for Graph Level tasks as we need to have a global representation of the graph. The principle of pooling is to reduce the size of the graph to obtain a small representation, while keeping essential information about the graph and neigborhoods of nodes. The same concept exists in classic convolutionnal networks for images. It consists in an additionnal layer in the network after the GNN layers that we presented earlier, where sets of nodes and edges will be summarized in a subset of nodes and edges. There are many types of pooling layers, but here are some regular pooling layers often used : 
 
- Global Average Pooling, where the average value of the set of nodes is taken to create a single value for this set. 

- Global Max Pooling, where the maximal value of the set is taken instead of the average.

**finish**

## II - Graph Neural Network Implementation

### 1 - Data and Kind of Predictions

We will work in this project with the ZINC open-source dataset. It contains a collection of 249,456 graph representations of comercially availaible chemical compounds. Using the PyTorch Gemoetric package enable us to download the preprocessed data already split into train, validation and test samples. 

- 220,011 observations for the Train Set
- 24,445 observations in the Validation Set 
- 5,000 observations in the Test Set

The data is structured with :

- A variable x, which describes the nodes of a graph
- A variable edge_index, to indicate the connections between nodes
- A variable edge_attr, which contains the type of bond (edge) between two nodes
- A variable y, which is our **target** variable, and measures the constrained solubility of the molecules. 

→ The constrained solubility of a molecule is given by the following formula : 

$$
y = log(p) - SAS - Cycles
$$

with : 

- *log(p)* being the logarithm of the water-octanol partition coefficient. This quantity is itself a measure of the relationship between fat solubility and water solubility of the molecule. 

- *SAS* is the Synthetic Accessibility Score, which measures how difficult a particular molecule is to synthesize.

- *Cycles* is the number of cyles with more than 6 atoms in the molecule.

The constrained solubility is usually used as an indicator of solubility of the molecule, and a measure of how drug-like a molecule is. Our task will be to predict this property using Graph Neural Networks.

You can find a general Data Exploration here : 
**LINK TO EXPLORATION.IPYNB** 

Here is two boxplots showing the distribution of our target variable, the second one not displaying extreme values in the data :

<p align="center">
<img src="images/report_imgs/boxes_w_outliers.png" alt="Image Alt Text" width="400"/>
<img src="images/report_imgs/boxes_logp.png" alt="Image Alt Text" width="400"/>
</p>

We can see that the majority of the graphs take values between -4 and 4. However, there is a subsequent number of outliers taking extremely low values below -20. **do i erase the outliers ?**

Then, using the package *networkx*, we are able to obtain a visual representation of our molecule graphs : 

<p align="center">
<img src="images/report_imgs/graph_ex.png" alt="Image Alt Text" width="400"/>
</p>

We can clearly see the structure of our molecule, with its nodes and edges. We are just unable to vizualise the type of bond between the nodes. **or color by edge_attr?**

An interesting thing to see with these vizualisitations is the  variability of the representation for a same graph, like in the following example : 

<p align="center">
<img src="images/report_imgs/graph01.png" alt="Image Alt Text" width="400"/>
</p>

If you look closely, you can see that these two plots show the exact same graph, with the same number of nodes and identical edges. However, the representation is completely different. This illustrates the visual representations of graphs property that we talked about above.

In summary, we are going to try to predict a chemical compound property of molecules from the ZINC dataset, which constitutes a Graph-Level prediction task.

