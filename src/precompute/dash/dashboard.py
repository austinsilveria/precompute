from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import pandas as pd

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

groups = list(set([md['artifact_group'] for md in artifact_metadata if 'artifact_group' in md]))
figure_names = groups + [md['name'] for md in artifact_metadata if 'artifact_group' not in md]

artifacts_by_figure = { fig_name: [] for fig_name in figure_names }
for md in artifact_metadata:
    if 'artifact_group' in md:
        artifacts_by_figure[md['artifact_group']].append(md)
    else:
        artifacts_by_figure[md['name']].append(md)

app.layout = html.Div([
    html.H1(children='Precompute Dashboard', className='h1'),
    dcc.Input(id='dummy-input', style={'display': 'none'}),
    html.Div([
        html.Div([
            dcc.Graph(id=f'graph-{name}', className='graph'),
            dcc.Markdown(id=f'markdown-{name}', className='markdown')
        ], className='graph-container') for name in figure_names
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
    dfs = []
    for artifacts in artifacts_by_figure.values():
        window = artifacts[0]['window-average'] if 'window-average' in artifacts[0] else 1
        artifact_data = [read_artifact_data(md['data_path']).to_pandas() for md in artifacts]
        combined = {
            artifact_data[0].columns[0]: artifact_data[0][artifact_data[0].columns[0]]
        }
        for i, ad in enumerate(artifact_data):
            for col in ad.columns[1:]:
                combined[f'{artifacts[i]["name"]}-{col}'] = ad[col].rolling(window=window).mean()
        dfs.append(pd.DataFrame(combined))
    dfs = [df.melt(id_vars=[df.columns[0]], var_name='var', value_name='val') for df in dfs]

    figures = []
    markdowns = []
    for i, artifact_md_list in enumerate(artifacts_by_figure.values()):
        df = dfs[i]
        md = artifact_md_list[0]

        log = True
        if md['visualization'] == 'linear-line':
            log = False
        fig = px.line(df, x=df.columns[0], y='val', color='var', title=figure_names[i], log_x=log, log_y=log, render_mode='svg')

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
            text=md['y_name'],
            xref="paper", yref="paper",
            x=-0.075, y=0.5,
            xanchor="right",
            showarrow=False,
            font=dict(size=14, color="#ecf0f1"),
            align="left",
        )
        figures.append(fig)

        markdown_text = md['description'] if 'description' in md else ''
        markdowns.append(markdown_text)
    
    return figures + markdowns

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')