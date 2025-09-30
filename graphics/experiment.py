from manim import *

class AnimatedBarPlot(Scene):
    def construct(self):
        # Data for the bar plot
        data = [3, 7, 5, 9, 2, 6]  # Example dataset
        labels = ["A", "B", "C", "D", "E", "F"]  # Labels for each bar

        # Create axes
        axes = Axes(
            x_range=[0, len(data), 1],
            y_range=[0, max(data) + 1, 1],
            axis_config={"color": BLUE},
        )
        axes_labels = axes.get_axis_labels(x_label="Category", y_label="Value")

        # Create bars
        bars = VGroup()  # Group to hold all bars
        for i, (value, label) in enumerate(zip(data, labels)):
            bar = Rectangle(
                height=value,
                width=0.5,
                fill_color=BLUE,
                fill_opacity=0.7,
                stroke_color=WHITE,
            )
            bar.next_to(axes.c2p(i, 0), DOWN, buff=0)  # Position bars on the axes
            bars.add(bar)

        # Add labels to bars
        bar_labels = VGroup()
        for i, (bar, label) in enumerate(zip(bars, labels)):
            label_text = Text(label, font_size=24).next_to(bar, DOWN)
            bar_labels.add(label_text)

        # Animation sequence
        self.play(Create(axes), Write(axes_labels))
        self.wait(1)
        self.play(LaggedStart(*[Create(bar) for bar in bars], lag_ratio=0.2))
        self.play(Write(bar_labels))
        self.wait(2)

# Run the scene
if __name__ == "__main__":
    scene = AnimatedBarPlot()
    scene.render()