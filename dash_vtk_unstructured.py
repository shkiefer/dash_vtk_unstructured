from flask import Flask
import numpy as np
import pandas as pd
import vtk
import pyvista as pv
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_vtk
from dash_vtk.utils import to_mesh_state
from dash_vtk.utils import presets

import base64
import io
from pathlib import Path

APP_ID = 'fea_vtk'


def toDropOption(name):
    return {"label": name, "value": name}

def ns_export_to_uGrid(n_ns_b64, e_ns_b64, e_data_b64=None):
    decoded_n = base64.b64decode(n_ns_b64.split(',')[1])
    decoded_e = base64.b64decode(e_ns_b64.split(',')[1])

    df_nodes = pd.read_csv(io.StringIO(decoded_n.decode('utf-8')), delim_whitespace=True, header=None, skiprows=1, names=['id', 'x', 'y', 'z'])

    df_nodes['id'] = df_nodes['id'].astype(int)
    df_nodes = df_nodes.set_index('id', drop=True)
    # fill missing ids in range as VTK uses position (index) to map cells to points
    df_nodes = df_nodes.reindex(np.arange(df_nodes.index.min(), df_nodes.index.max() + 1), fill_value=0)

    df_elems = pd.read_csv(io.StringIO(decoded_e.decode('utf-8')), skiprows=1, header=None, delim_whitespace=True, engine='python', index_col=None).sort_values(0)
    # order: 0: eid, 1: eshape, 2+: nodes
    df_elems.iloc[:, 0] = df_elems.iloc[:, 0].astype(int)
    n_nodes = df_elems.iloc[:, 1].map(lambda x: int(''.join(i for i in x if i.isdigit())))
    df_elems.insert(2, 'n_nodes', n_nodes)
    # fill missing ids in range as VTK uses position (index) to map data to cells
    df_elems = df_elems.set_index(0).reindex(np.arange(df_elems.iloc[:, 0].min(), df_elems.iloc[:, 0].max() + 1), fill_value=0)

    # mapping specific to Ansys Mechanical data
    vtk_shape_id_map = {
        'Tet4': vtk.VTK_TETRA,
        'Tet10': vtk.VTK_QUADRATIC_TETRA,
        'Hex8': vtk.VTK_HEXAHEDRON,
        'Hex20': vtk.VTK_QUADRATIC_HEXAHEDRON,
        'Tri6': vtk.VTK_QUADRATIC_TRIANGLE,
        'Quad8': vtk.VTK_QUADRATIC_QUAD,
        'Tri3': vtk.VTK_TRIANGLE,
        'Quad4': vtk.VTK_QUAD,
        'Wed15': vtk.VTK_QUADRATIC_WEDGE
    }
    df_elems['cell_types'] = df_elems.iloc[:, 0].map(lambda x: vtk_shape_id_map[x.strip()] if x.strip() in vtk_shape_id_map.keys() else np.nan)
    df_elems = df_elems.dropna(subset=['cell_types'], axis=0)
    # convert dataframes to vtk-desired format
    points = df_nodes[['x', 'y', 'z']].to_numpy()
    cell_types = df_elems['cell_types'].to_numpy()
    n_nodes = df_elems.iloc[:, 1].to_numpy()
    # subtract starting node id from all grid references in cells to avoid filling from 0 to first used node (in case mesh doesnt start at 1)
    p = df_elems.iloc[:, 2:-1].to_numpy() - df_nodes.index.min()
    # if you need to, re-order nodes here-ish
    a = np.hstack((n_nodes.reshape((len(n_nodes), 1)), p))

    # convert to flat numpy array
    cells = a.ravel()
    # remove nans (due to elements with different no. of nodes)
    cells = cells[np.logical_not(np.isnan(cells))]
    cells = cells.astype(int)
    # create grid
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    if e_data_b64:
        decoded_ed = base64.b64decode(e_data_b64.split(',')[1])
        df_elem_data = pd.read_csv(
            io.StringIO(decoded_ed.decode('utf-8')),
            delim_whitespace=True,
            header=None,
            skiprows=1,
            names=['id', 'val']
        )
        df_elem_data = df_elem_data.sort_values('id').set_index('id', drop=True)
        # fill missing ids in range as VTK uses position (index) to map data to cells
        df_elem_data = df_elem_data.reindex(np.arange(df_elems.index.min(), df_elems.index.max() + 1), fill_value=0.0)
        np_val = df_elem_data['val'].to_numpy()
        # assign data to grid with the name 'my_array'
        grid['my_array'] = np_val
        rng = [grid['my_array'].min(), grid['my_array'].max()]

    else:
        rng = [0., 1.]

    return grid, rng


layout = dbc.Container([
    html.H1('Unstructured Mesh Viewer'),
    dbc.Row([
        dbc.Col([
            html.H3('Step 1: Upload your mesh'),
            # upload for nodes
            dcc.Store(id=f'{APP_ID}_store', data=[True]),
            dcc.Upload(
                id=f'{APP_ID}_nodes_upload',
                multiple=False,
                children=[
                    dbc.FormGroup([
                        dbc.Label('Upload Node Mesh'),
                        dbc.Button('Upload Nodes', color='primary', block=True)
                    ])
                ]
            ),
            # upload for elems
            dcc.Upload(
                id=f'{APP_ID}_elems_upload',
                multiple=False,
                children=[
                    dbc.FormGroup([
                        dbc.Label('Upload Element Mesh'),
                        dbc.Button('Upload Elements', color='primary', block=True)
                    ])
                ]
            ),
            html.H3('Step 2: Upload Your Element Results'),
            # upload for elem data
            dcc.Upload(
                id=f'{APP_ID}_elem_data_upload',
                multiple=False,
                children=[
                    dbc.FormGroup([
                        dbc.Label('Upload Element Results Data'),
                        dbc.Button('Upload Results', color='primary', block=True)
                    ])
                ]
            ),
            html.H3('Step 3: Play with colors!'),
            dbc.FormGroup([
                dbc.Label('Color Map'),
                dcc.Dropdown(
                    id=f'{APP_ID}_dropdown_cs',
                    options=list(map(toDropOption, presets)),
                    value="erdc_rainbow_bright",
                ),
            ]),
            dbc.FormGroup([
                dbc.Label('Cell Value Threshold'),
                dcc.RangeSlider(
                    id=f'{APP_ID}_range_slider',
                    min=0.,
                    max=1.,
                    step=.01,
                    value=[0., 1.],
                    marks={
                        0.: '0%',
                        .25: '25%',
                        .50: '50%',
                        .75: '75%',
                        1.0: '100%'
                    },
                ),
                dbc.FormText('% of maximum value in data set')
            ]),
            dbc.Checklist(
                id=f'{APP_ID}_toggle_edges',
                options=[
                    {"label": "Show Element Edges", "value": 1}
                ],
                switch=True,
                value=[1]
            )
            ],
            width=4
        ),
        dbc.Col([
            html.Div(
                html.Div(
                    style={"width": "100%", "height": "100%"},
                    children=[
                        dash_vtk.View(
                            id=f'{APP_ID}_geom_rep',
                            children=None,
                            pickingModes=['hover'],
                        ),
                    ]
                ),
                style={"height": "100%"},
            ),
            html.Div(
                id=f'{APP_ID}_hover_info'
            )
        ]),
    ]),
],
    fluid=True,
    style={"height": "75vh"},
)


def add_dash(app):

    @app.callback(
        Output(f'{APP_ID}_geom_rep', 'children'),
        Output(f'{APP_ID}_store', 'data'),
        Output(f'{APP_ID}_nodes_upload', 'contents'),
        Output(f'{APP_ID}_elems_upload', 'contents'),
        Input(f'{APP_ID}_nodes_upload', 'contents'),
        Input(f'{APP_ID}_elems_upload', 'contents'),
        Input(f'{APP_ID}_elem_data_upload', 'contents'),
        Input(f'{APP_ID}_dropdown_cs', 'value'),
        Input(f'{APP_ID}_range_slider', 'value'),
        Input(f'{APP_ID}_toggle_edges', 'value'),
        State(f'{APP_ID}_store', 'data')
    )
    def fea_vtk_upload(node_contents, elem_contents, elem_data_contents, cs_name, rng_slide, toggle_edges, is_cleared):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trig_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # scenario 1: nodes data, but elems wasnt cleared (clears elems)
        if trig_id == f'{APP_ID}_nodes_upload' and not is_cleared[0]:
            return [dash.no_update, [True], dash.no_update, None]

        # scenario 2: elems data, but nodes wasnt cleared (clears nodes)
        if trig_id == f'{APP_ID}_elems_upload' and not is_cleared[0]:
            return [dash.no_update, [True], None, dash.no_update]

        # scenario 3: nodes data, but no elems data
        if trig_id == f'{APP_ID}_nodes_upload' and elem_contents is None:
            raise PreventUpdate

        # scenario 4: elems data, but no nodes data
        if trig_id == f'{APP_ID}_elems_upload' and node_contents is None:
            raise PreventUpdate

        # scenario 5: data for both mesh, but no results
        if all([node_contents is not None, elem_contents is not None, is_cleared[0], elem_data_contents is None]):
            uGrid, _ = ns_export_to_uGrid(node_contents, elem_contents)

            mesh_state = to_mesh_state(uGrid)

            return [
                dash_vtk.GeometryRepresentation(
                    [
                        dash_vtk.Mesh(state=mesh_state)
                    ],
                    property={"edgeVisibility": True, "opacity": 0.25, 'representation': 2}
                ),
                [False],
                dash.no_update,
                dash.no_update,
            ]

        # scenario 6: data for the whole shebang
        if all([node_contents is not None, elem_contents is not None, elem_data_contents is not None]):
            uGrid, rng = ns_export_to_uGrid(node_contents, elem_contents, elem_data_contents)

            mesh_state1 = to_mesh_state(uGrid)
            if rng_slide[0] == 0. and rng_slide[1] != 1.:
                # threshold to keep values below upper value of range slider
                thresh = uGrid.threshold_percent(rng_slide[1], invert=True)
            elif rng_slide[0] != 0. and rng_slide[1] == 1.:
                # threshold to keep values above lower value of range slider
                thresh = uGrid.threshold_percent(rng_slide[0])
            elif rng_slide[0] == 0. and rng_slide[1] == 1.:
                thresh = uGrid.copy()
            else:
                # threshold to keep values in range
                thresh = uGrid.threshold_percent(rng_slide)
            mesh_state2 = to_mesh_state(thresh, field_to_keep='my_array')

            if 1 in toggle_edges:
                show_edges = True
            else:
                show_edges = False

            return [
                [
                    # this is the ghost display to see the outline of the part
                    dash_vtk.GeometryRepresentation(
                        [
                            dash_vtk.Mesh(state=mesh_state1)
                        ],
                        property={"edgeVisibility": False, "opacity": .15},
                    ),
                    # this is the threshold solid elements within the range slider values
                    dash_vtk.GeometryRepresentation(
                        [
                            dash_vtk.Mesh(state=mesh_state2)
                        ],
                        property={"edgeVisibility": show_edges, "opacity": 1},
                        colorMapPreset=cs_name,
                        showCubeAxes=False,
                        colorDataRange=rng
                    ),
                ],
                [False],
                dash.no_update,
                dash.no_update,
            ]

    @app.callback(
        Output(f'{APP_ID}_hover_info', 'children'),
        Input(f'{APP_ID}_geom_rep', 'hoverInfo')
    )
    def fea_vtk_hover(hover_info):
        if hover_info is not None:
            hover_stuff = [[html.P(f'{k}: {hover_info[k]}'), html.Br()] for k in hover_info.keys()]
            # trick to flatten, not fastest but pretty fast
            return sum(hover_stuff, [])
        else:
            return ['']

    return app


if __name__ == '__main__':

    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
    ]

    server = Flask(__name__, static_folder=str(Path(__file__).parent.parent / "static/"))
    app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
    app.layout = layout
    app = add_dash(app)
    app.run_server(debug=True)
