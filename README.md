# graph-neural-networks

<p align="center">
<img src="images/molecule_AI.jpg" alt="Alt Text" width = "800">
</p>



    ├── images                      # Images for Report
    ├── Data_Exploration.ipynb      # Exploration Notebook 
    ├── GNN.ipynb                   # GNN model Notebook
    ├── Graph_Transformer.ipynb     # Graph Transformer Notebook
    ├── README.md              
    ├── REPORT.md                   # Comprehensive Report
    ├── requirements.txt            # Packages to install
    └── utils.py                    # Imports

This GitHub repository presents the Graph Neural Networks framework for Molecules property predictions. This project is part of my M2 diploma at the Toulouse School of Economics.

The goal is to predict the **Constrained Solubility** of molecules, defined by :

$$
y = log(P) - SAS - Cycles
$$

- $log(P)$ : Logarithm of the water-octanol partition coefficient (relation between fat sloubility and water solubility)

- $SAS$ : Synthetic Accessibility Score.

- $Cycles$ : Number of cycles containing 6 atoms or more

This property is a common indicator of how medcicinal a molecule is.

## GNN Architecture

<p align="center">
<img src="images/report_imgs/archi_gnn.png" alt="Alt Text" width = "650">
</p>

Result with 100 epochs :

- Train MAE : **0.6640**
- Validation MAE : **0.6788**
- Test MAE : **0.6773**

## Transformer Architecture

<p align="center">
<img src="images/report_imgs/archi_transformer.png" alt="Alt Text" width = "650">
</p>

Result with 100 epochs :

- Train MAE : **0.6446**
- Validation MAE : **0.6642**
- Test MAE : **0.6769**
