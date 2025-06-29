import os 
import sys 
import joblib
from manim import *
import torch.nn as nn 
from torch_geometric.data import Data, Batch, Dataset
import torch


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.model.graph_model import TAGNRegressor, FCNNRegressor

class AnimateModelBar(Scene): 
    def construct(self):
        device = torch.device("cpu") 
        ptr_dict = joblib.load(parent_dir+"/manimations/ptr_dict.pkl")
        ptr_dataset = ptr_dict["ptr_dataset"]
        scale_factor = ptr_dict["scale_factor"]
        
        model_id = 0
        match model_id: 
            case 0: 
                node_hidden_ptr_list = [32,16,8,4,3,2,3,4,8,16,32]
                k_hop_node_ptr = 1 # power flow equations.
                model_ptr = TAGNRegressor(node_in_features = ptr_dataset.x[0].shape[1], # 4
                                        node_hidden_features = node_hidden_ptr_list,
                                        node_out_features = ptr_dataset.x[0].shape[1], # 4
                                        k_hop_node = k_hop_node_ptr, 
                                        bias=True, 
                                        normalize=False).to(device)
                
                # load checkpoint 
                # checkpoint = torch.load(parent_dir+"/config/ptrNR_TAG_e600_nD_lr0.0001_d5000.pth", 
                #             weights_only=True)
                checkpoint = torch.load(parent_dir+"/config/ptrNR_TAG_e500_nD_lr0.0001_d4000_std1.pth", 
                            weights_only=True)
                

                # restore model state 
                model_ptr.load_state_dict(checkpoint["model_state_dict"])
                
            case 1: 
                hidden_feat_list = [32,16,8,4,3,2,3,4,8,16,32]
                model_ptr = FCNNRegressor(in_feat = ptr_dataset.x[0].shape[1], # 4
                                            hid_feat_list = hidden_feat_list,
                                            out_feat = ptr_dataset.x[0].shape[1]).to(device)

                # load checkpoint 
                checkpoint = torch.load(parent_dir+"/config/ptrFCNNR_e600_nD_lr0.0001_d5000.pth", 
                            weights_only=True)

                # restore model state 
                model_ptr.load_state_dict(checkpoint["model_state_dict"])

        total_params = sum(p.numel() for p in model_ptr.parameters() if p.requires_grad)
        print(f'Total number of parameters of pretraining model {model_ptr}: {total_params}')
        
        # generate tensor of inputs 
        input_tensor = torch.zeros((len(ptr_dataset),
                                    ptr_dataset[0].x.shape[0],
                                    ptr_dataset[0].x.shape[1]))
        output_tensor = torch.zeros_like(input_tensor)
        
        
        with torch.no_grad(): 
            for perm in range(len(ptr_dataset)): 
                input_tensor[perm] = ptr_dataset[perm].x 
                output_tensor[perm] = model_ptr(ptr_dataset[perm]) 
        
        input_tensor, output_tensor = input_tensor * scale_factor, output_tensor * scale_factor

        
        
        # print(input_tensor, output_tensor)
        feature_id = 3 
        y_min, y_max = np.floor(input_tensor[:,:,feature_id].min()), np.ceil(input_tensor[:,:,feature_id].max())
        
        data_list = [] 
        data_list = [
            (
                input_tensor[perm,:,feature_id].tolist(),
                output_tensor[perm,:,feature_id].tolist(),
                f"Sample {perm}"
            )
            for perm in range(len(ptr_dataset))
        ]

        lenx = ptr_dataset.x.shape[1]

        # Set the aspect ratio
        self.camera.frame_width = 42  # Set the width of the camera frame (x-axis size)
        self.camera.frame_height = 20  # Set the height of the camera frame (y-axis size)

        plane = NumberPlane(
                x_range = (0, lenx, 5),
                # y_range = (y_min, y_max, float(y_max - y_min)*0.25),
                y_range = (-3, 3, 0.8),
                x_length = 34,
                y_length = 12,
                axis_config={"include_numbers": True,"font_size": 72},
            ).to_edge(DOWN)

        plane.center()
      
        # labels 
        xlabel = Text("Buses", font_size=72).move_to(DOWN*8)
        match feature_id: 
            case 0: 
                ylabel = Text("Voltage Mag. (pu)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)
            case 1: 
                ylabel = Text("Voltage Angle (Deg)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)
            case 2: 
                ylabel = Text("Active Power (MW)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)
            case 3: 
                ylabel = Text("Reactive Power (MVAR)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)

        
        # legends 
        d_mod1 = Dot(radius=0.24).move_to(UP*8.5 + RIGHT*16)
        d_mod2 = Dot(radius=0.24).move_to(UP*8.5 + RIGHT*14)
        # model line 
        l_model = Line(d_mod1.get_center(), d_mod2.get_center(), color=GOLD_E).set_stroke(width = 7)

        d_wls1 = Dot(radius=0.24).move_to(UP*7 + RIGHT*16)
        d_wls2 = Dot(radius=0.24).move_to(UP*7 + RIGHT*14)
        # model line 
        l_wls = Line(d_wls1.get_center(), d_wls2.get_center(), color=BLUE).set_stroke(width = 7)
        
        match model_id: 
            case 0:
                title = Text("GNN Sample", font_size=72).move_to(UP*8 + LEFT)
            case 1: 
                title = Text("FCNN Sample", font_size=72).move_to(UP*8 + LEFT)

        leg1 = Text("Predictions", font_size=72).move_to(UP*8.5 + RIGHT*10)
        leg2 = Text("True Values", font_size=72).move_to(UP*7 + RIGHT*10)
        self.play(Write(title), Create(plane), Write(xlabel), Write(ylabel))
        self.wait(2)
        self.play(Write(leg1), Write(leg2))
        self.play(Create(l_model), Create(l_wls))

        for i, (input_data, output_data, title) in enumerate(data_list[::40]):
                          
            line_graph_input = plane.plot_line_graph(
                x_values = [i for i in range(lenx)],
                y_values = input_data,
                stroke_width = 12,
            ).set_stroke(color=BLUE, width=7)

            line_graph_output = plane.plot_line_graph(
                x_values = [i for i in range(lenx)],
                y_values = output_data,
                line_color=GOLD_E,
                stroke_width = 12,
            ).set_stroke(color=GOLD_E, width=7)

            sample_id = Text(str(i+1), font_size=72).move_to(UP*8.1+3*RIGHT)
            self.add(line_graph_input, line_graph_output, sample_id)
            self.wait(0.2)
            self.remove(line_graph_input, line_graph_output, sample_id)

  
if __name__ == "__main__":
    scene = AnimateModelBar()
    # scene.render()