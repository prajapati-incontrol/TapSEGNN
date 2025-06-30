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
├── config/                              # Configuration files and hyperparameter settings
   ├── config.yaml                       # Primary configuration for state estimation experiments
   ├── config_gan.yaml                   # Configuration parameters for GAN-based models and training
├── manimations/                         # Mathematical animations and visualisations using Manim
├── results/                             # Experiment outputs and analysis
   │                                     # Auto-generated Jupyter notebooks documenting each experiment run
   │                                     # Includes configuration, performance metrics and plots
├── src/                                 # Source files
   ├── dataset/                          
   │   ├── custom_dataset.py             # Custom dataset classes for power system data loading
   │                                     
   ├── model/                            
   │   ├── graph_model.py                # Graph neural network implementations 
   │                                     
   ├── training/                         
       ├── trainer.py                    # Main training orchestrator with loss functions and metrics
                                         # Supports both supervised and adversarial training modes
├── utils/                               
   ├── gen_utils.py                      # General-purpose utility functions
   │                                     
   ├── load_data_utils.py                # Data loading and preprocessing utilities
   │                                     
   ├── model_utils.py                    # Model-specific utility functions
   │                                     
   ├── plot_utils.py                     # Visualisation and plotting utilities
   │                                     
   ├── ppnet_utils.py                    # Pandapower network interface utilities
                                         
├── main.py                              # Main execution script and experiment orchestrator
│                                        
│                                        
├── requirements.txt                     # Python package dependencies and version specifications
│                                        
└── README.md                            # Project Documentation
                                         
```

_Note: Due to the use of private MV (Medium Voltage) network data, detailed analysis notebooks are not posted yet in this repository_

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repo, and create pull requests.

---






