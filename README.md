# TapSEGNN
A GAN-GNN framework for topology-aware state and transformer tap position estimation in unobservable distribution grids. This project is carried out as part of my Master's Thesis, in affiliation with TU Delft and Stedin B.V. It experiments with imputing missing measurements in MV networks using Generative Adversarial Networks conditioned on the network topology and synthetic power flow data--essentially reconstructing an unobservable electrical network. Furthermore, it independently proposes a TapSEGNN model utilising Graph and Simplicial Complex Neural Networks to estimate states and tap positions of MV/LV transformers, thereby improving situational awareness in the grid. 

## Visualisation of the State Estimation of the Trained Model on a real MV/LV network


### Installation

1. Clone this repository:

```bash
git clone https://github.com/prajapati-incontrol/gd4ps-jax.git
cd gd4ps-jax
```


2. Set up a virtual environment (recommended):


```bash
python -m venv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 📁 Project Structure

```
scnn-jax/
├── log/                    # log files
       ├── script_log.txt
├── data/                   # Sample transmission system data and graphs
├── src/                    # Source files 
       ├── dataset/         # Custom Dataset Object
       ├── model/           # SCNN models 
       ├── training/        # Trainer functions 
├── utils.py                # Topology and data processing utilities
├── main.py                 # Orchestrate everything
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---


## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repo, and create pull requests.

---






