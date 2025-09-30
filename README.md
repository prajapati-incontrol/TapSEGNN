# TapSEGNN
This project is carried out as part of my Master's Thesis, in affiliation with TU Delft and Stedin B.V. It experiments with imputing missing measurements in MV networks using Generative Adversarial Networks conditioned on the network topology and synthetic power flow data--essentially reconstructing an unobservable electrical network. Furthermore, it independently proposes a TapSEGNN model utilising Graph and Simplicial Complex Neural Networks to estimate states and tap positions of MV/LV transformers, thereby improving situational awareness in the grid. 

Checkout my thesis at: https://resolver.tudelft.nl/uuid:ed42f877-387a-481d-8a21-f80d2a872b4f


## Visualisation of the State Estimation of the Trained Model on a real MV/LV network


<p align="center">
  <img src="graphics/tapsegnn_v_pu_new.gif" width="400" />
  <img src="graphics/tapsegnn_v_deg_newq.gif" width="400" />
</p>


## Dataset Generation Summary 

![Dataset Generation Summary](graphics/FIG_dataset_generation_summary.jpeg)


### Installation

1. Clone this repository:

```bash
git clone https://github.com/prajapati-incontrol/TapSEGNN.git
cd TapSEGNN
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
TapSEGNN/
├── config/                              # Configuration files and hyperparameter settings
   ├── config.yaml                       # Primary configuration for state estimation experiments
   ├── config_gan.yaml                   # Configuration parameters for GAN-based models and training
├── graphics/                            # All graphics like mathematical animations and visualisations using Manim
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

_Note: Due to the use of private MV (Medium Voltage) network data, detailed analysis notebooks on open-source networks are currently being prepared and will be posted ASAP._

## The Way Forward 

1. Add a custom `collate` function to optimise the computation of transformer readout layers instead of forward pass in `src/model/graph_models.py`.
2. Speed up the calculation of Hodge-Laplacians in `utils/gen_utils.py`.
3. (Bit ambitious) Distributed computation using PyTorch Geometric algorithms. 



## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repo, and create pull requests.

---






