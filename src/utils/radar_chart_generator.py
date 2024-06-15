import textwrap
import time
import plotly.express as px
import plotly.graph_objects as go
from const import beautified_model_names, distinctive_colors


class RadarChartPlotter:
    def __init__(self):
        pass

    @staticmethod
    def customwrap(s, width=20):
        # remove reasoning from the model name
        s = s.replace(" Reasoning", "")
        return "<br>".join(textwrap.wrap(s, width=width))

    def plot_radar_chart(self, df, aspect_filter, included_models, file_path):
        df = df.loc[aspect_filter]
        fig = go.Figure()
        for column in included_models:
            values = df[column].values.tolist() + [df[column].values[0]]
            indices = df.index.tolist() + [df.index[0]]
            indices = [self.customwrap(index, 15) for index in indices]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=indices,
                fill='none',
                name=column,
                line=dict(width=5),
                marker=dict(size=12),
                mode='lines+markers'
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    tickfont=dict(
                        size=39,
                        family='Helvetica'
                    ),
                    titlefont=dict(
                        size=39,
                        family='Helvetica'
                    )
                ),
                angularaxis=dict(
                    tickfont=dict(
                        size=42,
                        family='Helvetica'
                    )
                )
            ),
            legend=dict(
                x=0.5,
                y=1.15,
                xanchor='center',
                orientation='h',
                font=dict(
                    size=39,
                    family='Helvetica'
                ),
                bgcolor='#e5ecf6',
                bordercolor='Black',
                borderwidth=2
            ),
            margin=dict(l=0, r=0)
        )
        fig1 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig1.write_image(file_path, width=2000, height=1500, scale=2)
        time.sleep(3)
        fig.write_image(file_path, width=2000, height=1500, scale=2)
        # purposely saving the figure twice to avoid MathJax watermark bug
