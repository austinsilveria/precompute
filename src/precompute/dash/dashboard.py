from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px

from precompute import list_artifacts, read_artifact_metadata, read_artifact_data

app = Dash(__name__)

# 1. Load all artifact metadata from the artifact cache
#   - name, visualization (e.g. 'line', 'histogram'), description, config
# 2. Render artifact browser
#   - table with selectable rows
#   - button to create a merge group (select multiple rows)
# 3. Render selected artifacts in the dashboard
#   - grid

# todo:
#   - add browser/selector
#   - support visualization types

artifact_metadata = [read_artifact_metadata(f) for f in list_artifacts()]
figure_names = [md['name'] for md in artifact_metadata]

app.layout = html.Div([
    html.H1(children='Precompute Dashboard', className='h1'),
    dcc.Input(id='dummy-input', style={'display': 'none'}),
    html.Div([
        html.Div([
            dcc.Graph(id=f'graph-{md["name"]}', className='graph'),
            dcc.Markdown(id=f'markdown-{md["name"]}', className='markdown')
        ], className='graph-container') for md in artifact_metadata
    ], className='graph-grid'),
    dcc.Interval(
        id='interval-component',
        interval=30*1000, # milliseconds
        n_intervals=0
    ),
])

@callback(
    [Output(f'graph-{name}', 'figure') for name in figure_names] + [Output(f'markdown-{name}', 'children') for name in figure_names],
    Input('interval-component', 'n_intervals'),
)
def update_graphs(n):
    dfs = [read_artifact_data(md['data_path']).to_pandas() for md in artifact_metadata]
    dfs = [df.melt(id_vars=[df.columns[0]], var_name='var', value_name='val') for df in dfs]

    figures = []
    markdowns = []
    for i, df in enumerate(dfs):
        fig = px.line(df, x=df.columns[0], y='val', color='var', title=figure_names[i], log_x=True, log_y=True, render_mode='svg')

        # Truncate long labels for rendering only
        labels = {col: col[:20] for col in df['var'].unique()}
        fig.for_each_trace(
            lambda t: t.update(name = labels[t.name], legendgroup = labels[t.name],)
        )

        fig.update_layout(
            xaxis=dict(title=dict(text=df.columns[0]), showgrid=False, showline=True, zeroline=False, mirror=False, ticks='outside', tickwidth=1, tickcolor='#7f8c8d'),
            yaxis=dict(title=dict(text=''), showgrid=False, showline=True, zeroline=False, mirror=False, ticks='outside', tickwidth=1, tickcolor='#7f8c8d'),
            plot_bgcolor='#2c3e50',
            paper_bgcolor='#2c3e50',
            font_color='#ecf0f1',
            margin=dict(l=120, r=20, t=100, b=40),
            legend_title_text='',
        )
        fig.add_annotation(
            text=artifact_metadata[i]['y_name'],
            xref="paper", yref="paper",
            x=-0.075, y=0.5,
            xanchor="right",
            showarrow=False,
            font=dict(size=14, color="#ecf0f1"),
            align="left",
        )
        figures.append(fig)

        markdown_text = artifact_metadata[i]['description'] if 'description' in artifact_metadata[i] else ''
        markdowns.append(markdown_text)
    
    return figures + markdowns

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')