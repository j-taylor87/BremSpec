import plotly.graph_objects as go


import plotly.io as pio
print(pio.templates)

# List of templates to try
templates = ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation']

# Iterate through each template and create a figure
for template in templates:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines+markers'))
    fig.update_layout(template=template, title=f"Template: {template}")
    fig.show()