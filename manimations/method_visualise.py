from manim import * 
import networkx as nx 
import sys 
import os 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.gen_utils import construct_hodge_laplacian

class AnimateGNN(Scene): 
    def construct(self): 
        vis_title = Text("Proposed Methodology", font_size=36).to_edge(UP)
        self.play(Write(vis_title))

        # one-line diagram of a 4 bus system
        # buses
        bus1 = Line(start = UP * 2, end = UP, color = GRAY, stroke_width = 10).move_to(LEFT * 6).set_z_index(2)
        bus2 = Line(start = UP * 2, end = UP, color = GRAY, stroke_width = 10).move_to(LEFT).set_z_index(2)
        bus3 = Line(start = UP * 2, end = UP, color = GRAY, stroke_width = 10).move_to(RIGHT * 3).set_z_index(2)
        final_pos_bus3 = LEFT * 4 + DOWN * 3
        bus4 = Line(start = DOWN*2, end = DOWN * 3, color = GRAY, stroke_width = 10).move_to(final_pos_bus3).set_z_index(2)

        # label the buses
        labbus_1 = Text("1").next_to(bus1, UP)
        labbus_2 = Text("2").next_to(bus2, UP)
        labbus_3 = Text("3").next_to(bus3, UP)
        labbus_4 = Text("4").next_to(bus4, UP)

        # create transmission lines
        line12 = Line(bus1.get_right(), bus2.get_left(), color = ORANGE).set_z_index(1)
        line23 = Line(bus2.get_right(), bus3.get_left(), color = GREEN).set_z_index(1)
        line14 = Line(bus1.get_right(), bus4.get_left(), color = ORANGE).set_z_index(1)
        line24 = Line(bus2.get_left(), bus4.get_right(), color = ORANGE).set_z_index(1)
        
        # trafo for 23
        trafo1 = Circle(radius=0.4, color = WHITE).move_to((3/4) * RIGHT).set_z_index(1)
        trafo2 = Circle(radius=0.4, color = WHITE).move_to((5/4) * RIGHT).set_z_index(1)
        
        self.play(DrawBorderThenFill(bus1), Write(labbus_1),
                  DrawBorderThenFill(bus2), Write(labbus_2),
                  DrawBorderThenFill(bus3), Write(labbus_3),
                  DrawBorderThenFill(bus4), Write(labbus_4))
        
        self.play(Create(line12),
                  Create(line23), 
                  Create(trafo1), 
                  Create(trafo2),
                  Create(line14),
                  Create(line24))
        

        # create transmission lines
        nline12 = Line(bus1.get_right(), bus2.get_left(), color = ORANGE).set_z_index(1)
        nline23 = Line(bus2.get_right(), bus3.get_left(), color = GREEN).set_z_index(1)
        nline14 = Line(bus1.get_right(), bus4.get_left(), color = ORANGE).set_z_index(1)
        nline24 = Line(bus2.get_left(), bus4.get_right(), color = ORANGE).set_z_index(1)

        node = Circle(radius = 0.2, color = GRAY, fill_opacity = 1)
        self.play(Transform(bus1,node.copy().move_to(bus1.get_center())),
                  Transform(bus2,node.copy().move_to(bus2.get_center())),
                  Transform(bus3,node.copy().move_to(bus3.get_center())),
                  Transform(bus4,node.copy().move_to(bus4.get_center())), run_time = 2)
        self.play(Transform(trafo1, nline23),
                  Transform(trafo2, nline23),
                  Transform(line12, nline12), 
                  Transform(line24, nline24),
                  Transform(line14, nline14),
                run_time = 2)

        self.play(FadeOut(labbus_1), FadeOut(labbus_2), FadeOut(labbus_3), FadeOut(labbus_4))

        # add node features 
        mscale = 0.5
        x1 = Matrix([["V_1"], ["\\theta_1"], ["P_1"], ["Q_1"]]).scale(mscale).next_to(bus1, UP)
        x2 = Matrix([["V_2"], ["\\theta_2"], ["P_2"], ["Q_2"]]).scale(mscale).next_to(bus2, UP)
        x3 = Matrix([["V_3"], ["\\theta_3"], ["P_3"], ["Q_3"]]).scale(mscale).next_to(bus3, UP)
        x4 = Matrix([["V_4"], ["\\theta_4"], ["P_4"], ["Q_4"]]).scale(mscale).next_to(bus4, UP)
        
        # add edge features 
        mescale = 0.5
        e12 = Matrix([["r_{12}"], ["x_{12}"], ["b_{12}"], ["g_{12}"], ["\\tau_{12}"], ["\\theta_{12}"]]).scale(mescale).next_to(nline12, 0.1*UP)
        e23 = Matrix([["r_{23}"], ["x_{23}"], ["b_{23}"], ["g_{23}"], ["\\tau_{23}"], ["\\theta_{23}"]]).scale(mescale).next_to(nline23, 0.1*UP)
        e14 = Matrix([["r_{14}"], ["x_{14}"], ["b_{14}"], ["g_{14}"], ["\\tau_{14}"], ["\\theta_{14}"]]).scale(mescale).next_to(nline14, 0.1*LEFT)
        e24 = Matrix([["r_{24}"], ["x_{24}"], ["b_{24}"], ["g_{24}"], ["\\tau_{24}"], ["\\theta_{24}"]]).scale(mescale).next_to(nline24, 0.1*RIGHT)

        self.play(Write(x1), Write(x2), Write(x3), Write(x4),
                  Write(e12), Write(e23), Write(e14), Write(e24))
        
        # Create a VGroup of all elements
        power_system_group = VGroup(bus1, bus2, bus3, bus4, 
                            line12, line23, line14, line24,
                            trafo1, trafo2, 
                            x1, x2, x3, x4,
                            e12, e23, e14, e24)

        # Scale and move the group to the left corner
        power_system_group  # UL means Upper Left

        # Play the transformation animation
        self.play(power_system_group.animate.scale(0.6).to_corner(LEFT * 0.5))

        # Collect the node-features 
        big_x = Matrix([
            ["V_1", "\\theta_1", "P_1", "Q_1"],
            ["V_2", "\\theta_2", "P_2", "Q_2"],
            ["V_3", "\\theta_3", "P_3", "Q_3"],
            ["V_4", "\\theta_4", "P_4", "Q_4"]
        ]).scale(0.6).move_to(UP * 1.5 + RIGHT * 3)  

        self.play(Transform(VGroup(x1, x2, x3, x4), big_x))

        # X = 
        x_label = MathTex("X = ").scale(0.8).next_to(big_x, LEFT)
        self.play(Write(x_label))

        # Collect the edge features 
        big_x1 = Matrix([
            ["r_{12}", "x_{12}", "b_{12}", "g_{12}", "\\tau_{12}", "\\theta_{12}"],
            ["r_{23}", "x_{23}", "b_{23}", "g_{23}", "\\tau_{23}", "\\theta_{23}"],
            ["r_{14}", "x_{14}", "b_{14}", "g_{14}", "\\tau_{14}", "\\theta_{14}"],
            ["r_{24}", "x_{24}", "b_{24}", "g_{24}", "\\tau_{24}", "\\theta_{24}"],
        ]).scale(0.6).move_to(DOWN * 1.5 + RIGHT * 3.7)

        self.play(Transform(VGroup(e12, e23, e14, e24), big_x1))

        # X^1 = 
        x1_label = MathTex("X^1 = ").scale(0.8).next_to(big_x1, LEFT)
        self.play(Write(x1_label))

           
        graph_n = VGroup(bus1, bus2, bus3, bus4, 
                            line12, line23, line14, line24,
                            trafo1, trafo2)

        # move to top and further scale down 
        self.play(graph_n.animate.scale(0.6).move_to(LEFT * 5 + UP * 2))

        # set color to WHITE 
        self.play(line12.animate.set_color(WHITE),
          line23.animate.set_color(WHITE),
          line14.animate.set_color(WHITE),
          line24.animate.set_color(WHITE), 
          trafo1.animate.set_color(WHITE),
          trafo2.animate.set_color(WHITE))
        
        # for directed graph 
        tl = 0.2
        arrow12 = Arrow(bus1.get_right(), bus2.get_left(), color=WHITE, buff=0.,tip_length = tl).set_z_index(1)
        arrow23 = Arrow(bus2.get_right(), bus3.get_left(), color=WHITE, buff=0.,tip_length = tl).set_z_index(1)
        arrow14 = Arrow(bus1.get_right(), bus4.get_left(), color=WHITE, buff=0.,tip_length = tl).set_z_index(1)
        arrow24 = Arrow(bus4.get_right(), bus2.get_left(), color=WHITE, buff=0.,tip_length = tl).set_z_index(1)

        
        bus1_c, bus2_c, bus3_c, bus4_c = [bus.copy() for bus in [bus1, bus2, bus3, bus4]]
        
        graph_e = VGroup(bus1_c, bus2_c, bus3_c, bus4_c, 
                         arrow12, arrow23, arrow14, arrow24).move_to(LEFT * 5 + DOWN * 2)
        
        self.play(Create(graph_e))

        # Now transform the undirected graph to graph adjacency 
        # collect the adjacency matrix 
        big_s = Matrix([
            [1,0,0,1],
            [1,0,1,1],
            [0,1,0,0],
            [1,1,0,0]
        ]).scale(0.6).move_to(bus4.get_center() + RIGHT * 1.5)

        self.play(Transform(graph_n, big_s))
        s_label = MathTex("S = ").scale(0.6).next_to(big_s, LEFT*1.5)
        self.play(Write(s_label))
        
        # animate dependency 
        d_rt = 0.07

        # show node-edge dependency in graph object 
        for _ in range(0):
            self.play(bus1_c.animate.set_color(GREEN), arrow12.animate.set_color(GREEN), arrow14.animate.set_color(GREEN), run_time = d_rt)
            # self.wait(0.1)
            self.play(bus1_c.animate.set_color(WHITE), arrow12.animate.set_color(WHITE), arrow14.animate.set_color(WHITE), run_time = d_rt)

        # show edge-triangle dependency 
        triangle_patch = Polygon(
            bus1_c.get_center(), bus2_c.get_center(), bus4_c.get_center(),
            color=YELLOW, fill_opacity=1  # Semi-transparent
        )
        
        for _ in range(0):
            self.play(arrow12.animate.set_color(RED),arrow14.animate.set_color(RED),arrow24.animate.set_color(RED), run_time = d_rt)
            # self.wait(0.2)
            self.play(arrow12.animate.set_color(WHITE),arrow14.animate.set_color(WHITE),arrow24.animate.set_color(WHITE), run_time = d_rt)

            self.play(FadeIn(triangle_patch), run_time = d_rt)
            # self.wait(0.2)
            self.play(FadeOut(triangle_patch), run_time = d_rt)

        # hodge-laplacians 
        G = nx.DiGraph()

        edges = [[1,2],[2,3],[1,4],[4,2]]

        G.add_edges_from(edges)

        (L_l, L_u), (B_1, B_2) = construct_hodge_laplacian(G)
        print(L_l)
        print(L_u)

        # matrix objects 
        b1_obj= Matrix(B_1).scale(0.6).move_to(bus4_c.get_center() + RIGHT * 1.7)
        b2_obj = Matrix(B_2).scale(0.6).move_to(bus4_c.get_center() + RIGHT * 5.1)
        b1b2 = VGroup(b1_obj, b2_obj)
        ll_obj, lu_obj = Matrix(L_l), Matrix(L_u)
        
        # transform graph_e to incidence matrices 
        b1_label = MathTex("B_1 = ").scale(0.6).next_to(b1_obj, LEFT)
        b2_label = MathTex("B_2 = ").scale(0.6).next_to(b2_obj, LEFT)
        
        self.play(Transform(graph_e, b1b2), Write(b1_label), Write(b2_label))

        # Define row and column indices
        # Add row indices
        row_indices = [Text(str(i+1)).scale(0.6).move_to(bus4.get_center() + RIGHT * 1.5).next_to(big_s.get_rows()[i], LEFT * 1.2) for i in range(4)]
        # Add column indices
        column_indices = [Text(str(j+1)).scale(0.6).move_to(bus4.get_center() + RIGHT * 1.5).next_to(big_s.get_columns()[j], UP * 1.2) for j in range(4)]

        r_bigs = VGroup(*row_indices).next_to(big_s, LEFT)
        c_bigs = VGroup(*column_indices).next_to(big_s, UP)
        self.play(Write(r_bigs), Write(c_bigs))

        # do the same for B_1, B_2 

        # illustrate the L_1, L_2 

        # focus on S, L_1, L_2 , X, X^1, \tau_id 

        # make a box as input 

        # X^o, 
        


        








        



if __name__ == "__main__":
    scene = AnimateGNN()