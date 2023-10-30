import dash
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import alphashape
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import glob2 as glob
from sklearn.decomposition import PCA
import open3d as o3d
from pyntcloud import PyntCloud
from sklearn.cluster import KMeans
import itertools
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree
import pickle
import os.path
import os
import networkx as nx
import scipy
from scipy.optimize import fsolve
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from scipy.integrate import quad


def cart_to_sphere(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    ptsnew[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    ptsnew[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2]) # for elevation angle defined from Z-axis down
    ptsnew[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew

def sphereFit_fixed_r(spX,spY,spZ,r0):

    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)

    xyz_array = np.zeros((len(spX), 3))
    xyz_array[:, 0] = spX
    xyz_array[:, 1] = spY
    xyz_array[:, 2] = spZ
    c0 = np.mean(xyz_array, axis=0)
    c0[2] = -r0
    def ob_fun(c0, xyz=xyz_array, r=r0):
        res = np.sqrt((xyz[:, 0]-c0[0])**2 + (xyz[:, 1]-c0[1])**2 + (xyz[:, 2]-c0[2])**2) - r
        return res

    C = scipy.optimize.least_squares(ob_fun, c0, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]))

    return r0, C.x[0], C.x[1], C.x[2]

def calculate_distance_metrics(fin_tip_point, point_stat_raw, AG):
    if fin_tip_point:
        cm_vec = [fin_tip_point[0]["x"], fin_tip_point[0]["y"], fin_tip_point[0]["z"]]
    else:
        cm_vec = np.mean(point_stat_raw.iloc[:, 0:3], axis=0)

    xyz_array_cm = point_stat_raw.iloc[:, 0:3].to_numpy() - cm_vec
    # calculate higher-order position stats
    xyz_df = pd.DataFrame(xyz_array_cm, columns=["X", "Y", "Z"])
    sph_arr = cart_to_sphere(xyz_array_cm)
    sph_df = pd.DataFrame(sph_arr, columns=["r", "lat", "lon"])
    for i in product(xyz_df, xyz_df, repeat=1):
        name = "*".join(i)
        xyz_df[name] = xyz_df[list(i)].prod(axis=1)

    # Graph-based stuff
    if fin_tip_point:
        fin_tip_ind = fin_tip_point[0]["pointNumber"]
    else:
        fin_tip_ind = []
    tip_dists = calculate_network_distances(fin_tip_ind, AG)
    
    # combine
    point_stat_df = pd.concat((sph_df, xyz_df), axis=1)
    point_stat_df["fin_tip_dists"] = tip_dists

    return point_stat_df

def calculate_network_distances(point, G):

    dist_array = np.empty((len(G)))
    if point:
        graph_distances = nx.single_source_dijkstra_path_length(G, str(point))

        for d in range(len(G)):
            try:
                dist_array[d] = graph_distances[str(d)]
            except:
                dist_array[d] = -1

        max_dist = np.max(dist_array)
        dist_array[np.where(dist_array == -1)[0]] = max_dist + 1
    else:
        dist_array[:] = 100

    return dist_array

def calculate_adjacency_graph(df, k_nn = 5):
    
    # calculate KD tree and use this to determine k nearest neighbors for each point
    xyz_array = df[["X", "Y", "X"]]
    # print(xyz_array)
    tree = KDTree(xyz_array)

    # get nn distances
    nearest_dist, nearest_ind = tree.query(xyz_array, k=k_nn + 1)

    # find average distance to kth closest neighbor
    mean_nn_dist_vec = np.mean(nearest_dist, axis=0)
    nn_thresh = mean_nn_dist_vec[k_nn]

    # iterate through points and build adjacency network
    G = nx.Graph()

    for n in range(xyz_array.shape[0]):
        node_to = str(n)
        nodes_from = [str(f) for f in nearest_ind[n, 1:]]
        weights_from = [w for w in nearest_dist[n, 1:]]

        # add node
        G.add_node(node_to)
        # only allow edges that are within twice the average k_nn distance
        allowed_edges = np.where(weights_from <= 2 * nn_thresh)[0]
        for a in allowed_edges:
            G.add_edge(node_to, nodes_from[a], weight=weights_from[a])

    return G

def calculate_point_cloud_stats(df):
    ########################
    # convert to point cloud
    xyz_array = np.asarray(df[["X", "Y", "Z"]]).copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)

    # generate features to train classifier
    cloud = PyntCloud.from_instance("open3d", pcd)

    k_vec = [101]

    for k in range(len(k_vec)):
        k_neighbors = cloud.get_neighbors(k=k_vec[k])

        ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        # cloud.add_scalar_field("eigen_decomposition", k_neighbors=k_neighbors)

        cloud.add_scalar_field("curvature", ev=ev)
        # cloud.add_scalar_field("anisotropy", ev=ev)
        cloud.add_scalar_field("eigenentropy", ev=ev)
        # cloud.add_scalar_field("eigen_sum", ev=ev)
        # cloud.add_scalar_field("linearity", ev=ev)
        # cloud.add_scalar_field("omnivariance", ev=ev)
        cloud.add_scalar_field("planarity", ev=ev)
        cloud.add_scalar_field("sphericity", ev=ev)

    point_stat_raw = cloud.points
    point_stat_raw = point_stat_raw.filter(regex="planar|curv")
    point_stat_raw = pd.concat((cloud.points.iloc[:, 0:3], point_stat_raw), axis=1)
    return point_stat_raw


def call_random_forest(features_train, features_all, fin_class, model=None):

    # print(features_train.head(2))
    if model == None:
        model = RandomForestClassifier(random_state=0)

    # features_train = features_train[:, np.newaxis]
    model.fit(features_train, fin_class.ravel())
    predictions = model.predict(features_all)

    return predictions, model
#
# def polyval2d(x, y, m):
#     order = int(np.sqrt(len(m))) - 1
#     ij = itertools.product(range(order + 1), range(order + 1))
#     z = np.zeros_like(x)
#     for a, (i, j) in zip(m, ij):
#         z += a * x ** i * y ** j
#     return z
#
# def polyfit2d(x, y, z, order=2):
#     ncols = (order + 1) ** 2
#     G = np.zeros((x.size, ncols))
#     ij = itertools.product(range(order + 1), range(order + 1))
#     for k, (i, j) in enumerate(ij):
#         G[:, k] = x ** i * y ** j
#     m, _, _, _ = np.linalg.lstsq(G, z)
#     return m

def visualize_atlas(dataRoot):

    global nucleus_df

    initialize_label_fields = False
    if os.path.isfile(os.path.join(dataRoot, "morph_df_curated.csv")):
        nucleus_df = pd.read_csv(os.path.join(dataRoot, "morph_df_curated.csv"), index_col=0)
    elif os.path.isfile(os.path.join(dataRoot, "morph_df.csv")):
        nucleus_df = pd.read_csv(os.path.join(dataRoot, "morph_df.csv"), index_col=0)
        initialize_label_fields = True
    else:
        raise Exception("No nucleus dataset found.")

    well_time_index = np.unique(nucleus_df["file"])
    if initialize_label_fields:
        # get list of unique well-time point combinations and add to dataframe
        for w, well_time in enumerate(well_time_index):
            nucleus_df.loc[nucleus_df["file"] == well_time, "well_time_index"] = int(w)

        # initialize label and label indicator columns
        nucleus_df["is_labeled_flag"] = 0
        nucleus_df["labels_assigned"] = -1
        # nucleus_df["special_labels_assigned"] = -1
        nucleus_df["labels_rf"] = -1

    def load_nucleus_dataset(well_time_index):

        modelPath = dataRoot + 'model.sav'
        modelDataPath = dataRoot + 'model_data.csv'

        model = []
        if os.path.isfile(modelPath):
            model = pickle.load(open(modelPath, 'rb'))

        model_data = []
        if os.path.isfile(modelDataPath):
            model_data = pd.read_csv(modelDataPath)

        df = nucleus_df.loc[nucleus_df["file"] == well_time_index, :]

        # Is should change this to store all of these attributes within a single dictionary
        base_points_prev = []
        fin_points_prev = []
        other_points_prev = []
        class_predictions_curr = []
        fin_tip_prev = []

        base_indices = np.where(df["labels_assigned"] == 0)[0]
        if np.any(base_indices):
            base_points_prev = [{"ind": i, "x": df.loc[i, "X"], "y": df.loc[i, "Y"], "z": df.loc[i, "z"]} for i in base_indices]

        other_indices = np.where(df["labels_assigned"] == 1)[0]
        if np.any(other_indices):
            other_points_prev = [{"ind": i, "x": df.loc[i, "X"], "y": df.loc[i, "Y"], "z": df.loc[i, "z"]} for i in
                                other_indices]

        fin_indices = np.where((df["labels_assigned"] == 2) | (df["labels_assigned"] == 3))[0]
        if np.any(fin_indices):
            fin_points_prev = [{"ind": i, "x": df.loc[i, "X"], "y": df.loc[i, "Y"], "z": df.loc[i, "z"]} for i in
                                 fin_indices]

        fin_tip_indices = np.where(df["labels_assigned"] == 3)[0]
        if np.any(fin_tip_indices):
            fin_tip_prev = [{"ind": i, "x": df.loc[i, "X"], "y": df.loc[i, "Y"], "z": df.loc[i, "z"]} for i in
                               fin_tip_indices]

        if np.min(df["labels_rf"]) >= 0:
            class_predictions_curr = df["labels_rf"].to_numpy()


        return {"df": df, "class_predictions_curr": class_predictions_curr, "fin_points_prev": fin_points_prev,
                "base_points_prev": base_points_prev, "other_points_prev": other_points_prev,
                "fin_tip_prev": fin_tip_prev}


    df_dict = load_nucleus_dataset(well_time_index=0)

    global base_points_prev, fin_points_prev, other_points_prev, fin_tip_prev, class_predictions_curr, pca_fin_prev, fin_surf_prev

    df = df_dict["df"]
    base_points_prev = df_dict["base_points_prev"]
    other_points_prev = df_dict["other_points_prev"]
    fin_points_prev = df_dict["fin_points_prev"]
    fin_tip_prev = df_dict["fin_tip_prev"]
    class_predictions_curr = df_dict["class_predictions_curr"]

    ########################
    # App
    app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
    global init_toggle
    init_toggle = True
    def create_figure(df):
        # calculate axis limits
        xmin = np.min(df["X"].iloc[:]) + 5
        xmax = np.max(df["X"].iloc[:]) - 5

        ymin = np.min(df["Y"].iloc[:]) - 5
        ymax = np.max(df["Y"].iloc[:]) + 5

        zmin = np.min(df["Z"].iloc[:]) - 3
        zmax = np.max(df["Z"].iloc[:]) + 3

        r_vec = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
        fig = px.scatter_3d(df, x="X", y="Y", z="Z", opacity=0.4, color=r_vec, color_continuous_scale='ice')
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(coloraxis_showscale=False)
        # fig.update_coloraxes(showscale=False)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[xmin, xmax], ),
                yaxis=dict(range=[ymin, ymax], ),
                zaxis=dict(range=[zmin, zmax], autorange="reversed"),
                aspectratio=dict(x=1, y=1, z=0.5)))

        return fig

    f = create_figure(df)

    app.layout = html.Div([
                        html.Button('Clear Selection', id='clear'),
                        html.Button('Save', id='save-button'),
                        html.Button('Fit Random Forest', id='calc-button'),
                        html.P(id='save-button-hidden', style={'display': 'none'}),
                        dcc.Graph(id='3d_scat', figure=f),
                        html.Div(id='df_list', hidden=True),
                        html.Div(id='other_points', hidden=True),
                        html.Div(id='base_points', hidden=True),
                        html.Div(id='fin_points', hidden=True),
                        html.Div(id='fin_tip_point', hidden=True),
                        html.Div(id='pfin_nuclei', hidden=True),

                        html.Div([
                            dcc.Dropdown(well_time_index, well_time_index[0], id='dataset-dropdown'),
                            html.Div(id='dd-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(["Fin Tip", "Pec Fin", "Base Surface", "Other"], "Fin Tip", id='class-dropdown'),
                            html.Div(id='class-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'})
                        ]
                        )

    @app.callback(
        Output('dd-output-container', 'children'),
        Input('dataset-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(
        Output('class-output-container', 'children'),
        Input('class-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback([Output('base_points', 'children'),
                   Output('other_points', 'children'),
                   Output('fin_points', 'children'),
                   Output('fin_tip_point', 'children')],
                    [Input('3d_scat', 'clickData'),
                     Input('clear', 'n_clicks'),
                     Input('class-output-container', 'children'),
                     Input('dd-output-container', 'children')],
                    [State('base_points', 'children'),
                     State('other_points', 'children'),
                     State('fin_points', 'children'),
                     State('fin_tip_point', 'children')
                     ])

    def select_point(clickData, n_clicks, class_val, fileName, base_points, other_points, fin_points, fin_tip_point):
        ctx = dash.callback_context
        ids = [c['prop_id'] for c in ctx.triggered]
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        # toggle_val = toggle_switch == "Click to select Pec Fin points."

        df_dict = load_nucleus_dataset(fileName)

        base_points_prev = df_dict["base_points_prev"]
        fin_points_prev = df_dict["fin_points_prev"]
        other_points_prev = df_dict["other_points_prev"]
        fin_tip_prev = df_dict["fin_tip_prev"]

        if base_points and ('dd-output-container' not in changed_id):
            base_results = json.loads(base_points)
        else:
            base_results = []

        if other_points and ('dd-output-container' not in changed_id):
            other_results = json.loads(other_points)
        else:
            other_results = []

        if fin_points and ('dd-output-container' not in changed_id):
            fin_results = json.loads(fin_points)
        else:
            fin_results = []

        if fin_tip_point and ('dd-output-container' not in changed_id):
            fin_tip_point = json.loads(fin_tip_point)
        else:
            fin_tip_point = []

        global init_toggle # NL: does this do anything?

        if ('dd-output-container' in changed_id) | init_toggle:
            if len(base_points_prev) > 0:
                for p in base_points_prev:
                    if p not in base_results: # NL: Does this do anything? why would there be base results already?
                        base_results.append(p)

            if len(fin_points_prev) > 0:
                for p in fin_points_prev:
                    if p not in fin_results:
                        fin_results.append(p)

            if len(other_points_prev) > 0:
                for p in other_points_prev:
                    if p not in other_results:
                        other_results.append(p)
            if fin_tip_prev:
                fin_tip_point = [fin_tip_prev[0]]


        if '3d_scat.clickData' in ids:
            if class_val == "Base Surface":
                xyz_nf = np.round([[base_results[i]["x"], base_results[i]["y"], base_results[i]["z"]] for i in
                                   range(len(base_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_nf:
                        base_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_nf == xyz_p)
                        base_results.pop(rm_ind[0][0])

            elif class_val == "Pec Fin":
                xyz_f = np.round([[fin_results[i]["x"], fin_results[i]["y"], fin_results[i]["z"]] for i in range(len(
                    fin_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_f:
                        fin_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_f == xyz_p)
                        fin_results.pop(rm_ind[0][0])

            elif class_val == "Other":
                xyz_o = np.round([[other_results[i]["x"], other_results[i]["y"], other_results[i]["z"]] for i in range(len(
                    other_results))], 3)
                for p in clickData['points']:
                    xyz_p = np.round([p["x"], p["y"], p["z"]], 3)
                    if [xyz_p] not in xyz_o:
                        other_results.append(p)
                    else:
                        # if selected point is in list, then lets unselect
                        rm_ind = np.where(xyz_o == xyz_p)
                        other_results.pop(rm_ind[0][0])

            elif class_val == "Fin Tip":
                fin_tip_point = clickData['points']
        if ('clear' in changed_id):
            base_results = []
            fin_results = []
            other_results = []
            fin_tip_point = []

        # This guards against weird error where click sometimes does not store point coordinates
        if fin_tip_point:
            if "x" not in fin_tip_point[0]:
                fin_tip_point = []

        base_results = json.dumps(base_results)
        fin_results = json.dumps(fin_results)
        other_results = json.dumps(other_results)
        fin_tip_point = json.dumps(fin_tip_point)

        init_toggle = False #NL: does this do anything?

        return base_results, other_results, fin_results, fin_tip_point

    @app.callback([Output('3d_scat', 'figure'),
                   Output('pfin_nuclei', 'children')],
                [Input('pfin_nuclei', 'children'),
                 Input('base_points', 'children'),
                 Input('other_points', 'children'),
                 Input('fin_points', 'children'),
                 Input('fin_tip_point', 'children'),
                 Input('calc-button', 'n_clicks'),
                 Input('dd-output-container', 'children')])

    def chart_3d(class_predictions_in, base_points, other_points, fin_points, fin_tip_point,
                 n_clicks, well_time_index):

        global f

        # check to see which values have changed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df_dict = load_nucleus_dataset(well_time_index)
        df = df_dict["df"]

        class_predictions_curr = df_dict["class_predictions_curr"]

        f = create_figure(df)

        f.update_layout(uirevision="Don't change")

        base_points = json.loads(base_points) if base_points else []
        fin_points = json.loads(fin_points) if fin_points else []
        other_points = json.loads(other_points) if other_points else []
        fin_tip_point = json.loads(fin_tip_point) if fin_tip_point else []
        if fin_tip_point:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[fin_tip_point[0]["x"]],
                    y=[fin_tip_point[0]["y"]],
                    z=[fin_tip_point[0]["z"]],
                    marker=dict(
                        color='black',
                        size=8,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )
        if base_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in base_points],
                    y=[p['y'] for p in base_points],
                    z=[p['z'] for p in base_points],
                    marker=dict(
                        color='crimson',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        if fin_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in fin_points],
                    y=[p['y'] for p in fin_points],
                    z=[p['z'] for p in fin_points],
                    marker=dict(
                        color='lightgreen',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        if other_points:
            f.add_trace(
                go.Scatter3d(
                    mode='markers',
                    x=[p['x'] for p in other_points],
                    y=[p['y'] for p in other_points],
                    z=[p['z'] for p in other_points],
                    marker=dict(
                        color='azure',
                        size=7,
                        line=dict(
                            color='black',
                            width=4
                        )
                    ),
                    showlegend=False
                )
            )

        # global class_predictions_curr
        if 'calc-button' in changed_id:
            if base_points and fin_points and other_points:

                # match clicked points to ref points using xyz coordinates
                xyz_click = np.empty((len(other_points) + len(base_points) + len(fin_points), 3))
                for o, op in enumerate(other_points):
                    xyz_click[o, :] = np.round([op["x"], op["y"], op["z"]], 3)

                for b, bp in enumerate(base_points):
                    xyz_click[b+len(other_points), :] = np.round([bp["x"], bp["y"], bp["z"]], 3)

                for fn, fp in enumerate(fin_points):
                    xyz_click[fn+len(other_points)+len(base_points), :] = np.round([fp["x"], fp["y"], fp["z"]], 3)

                dist_mat = distance_matrix(xyz_click, np.asarray(df[["X", "Y", "Z"]]))
                lb_indices = np.argmin(dist_mat, axis=1)

                # calculate point cloud stats
                point_stat_raw = calculate_point_cloud_stats(df)

                ################
                # calculate network-based stats
                G = calculate_adjacency_graph(df)

                point_stat_df = calculate_distance_metrics(fin_tip_point, point_stat_raw, G)

                # generate class vec
                fin_class_vec = np.zeros((len(base_points) + len(other_points) + len(fin_points), 1))
                fin_class_vec[len(other_points):, 0] = 1  # base=1
                fin_class_vec[len(base_points) + len(other_points):, 0] = 2  # fin=2

                df_lb = point_stat_df.iloc[lb_indices]

                class_predictions, model = call_random_forest(df_lb, point_stat_df, fin_class_vec)

            else:
                class_predictions = np.zeros((df.shape[0],))

        elif (class_predictions_in != None) and ('dd-output-container' not in changed_id):
            class_predictions = class_predictions_in
        else:
            class_predictions = class_predictions_curr

        pec_fin_nuclei = np.where(np.asarray(class_predictions) == 2)[0]
        base_nuclei = np.where(np.asarray(class_predictions) == 1)[0]
        other_nuclei = np.where(np.asarray(class_predictions) == 0)[0]

        ################
        # Plot predictions
        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[df["X"].iloc[p] for p in pec_fin_nuclei],
                y=[df["Y"].iloc[p] for p in pec_fin_nuclei],
                z=[df["Z"].iloc[p] for p in pec_fin_nuclei],
                marker=dict(
                    color='lightgreen',
                    opacity=0.25,
                    size=5),
                showlegend=False
            )
        )

        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[df["X"].iloc[p] for p in base_nuclei],
                y=[df["Y"].iloc[p] for p in base_nuclei],
                z=[df["Z"].iloc[p] for p in base_nuclei],
                marker=dict(
                    color='coral',
                    opacity=0.5,
                    size=5),
                showlegend=False
            )
        )

        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[df["X"].iloc[p] for p in other_nuclei],
                y=[df["Y"].iloc[p] for p in other_nuclei],
                z=[df["Z"].iloc[p] for p in other_nuclei],
                marker=dict(
                    color='azure',
                    opacity=0.5,
                    size=5),
                showlegend=False
            )
        )

        class_predictions = np.asarray(class_predictions)
        class_predictions = json.dumps(class_predictions.tolist())

        return f, class_predictions

    @app.callback(
        Output('save-button-hidden', 'children'),
        [Input('save-button', 'n_clicks'),
            Input('pfin_nuclei', 'children'),
            Input('base_points', 'children'),
            Input('fin_points', 'children'),
            Input('other_points', 'children'),
            Input('fin_tip_point', 'children'),
            Input('dd-output-container', 'children')])

    def clicks(n_clicks, class_predictions, base_points, fin_points, other_points,
                                                                fin_tip_point, fileName):

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'save-button' in changed_id:

            # add fields to main data freame
            df_dict = load_nucleus_dataset(fileName)
            df = df_dict["df"]
            class_predictions = json.loads(class_predictions)
            # add nucleus class predictions
            # class_predictions = class_predictions
            if len(class_predictions) == df.shape[0]:
                df.loc[:, "labels_rf"] = class_predictions

            base_points = json.loads(base_points)
            other_points = json.loads(other_points)
            fin_points = json.loads(fin_points)
            fin_tip_point = json.loads(fin_tip_point)

            for b in base_points:
                df.loc[b["pointNumber"], "labels_assigned"] = 0
            for o in other_points:
                df.loc[o["pointNumber"], "labels_assigned"] = 1
            for p in fin_points:
                df.loc[p["pointNumber"], "labels_assigned"] = 2
            for ft in fin_tip_point:
                df.loc[ft["pointNumber"], "labels_assigned"] = 3


            for col in nucleus_df.columns:
                nucleus_df.loc[nucleus_df["file"] == fileName, col] = df.loc[:, col]


            nucleus_df.to_csv(os.path.join(dataRoot, "morph_df_curated.csv"))
            # save
            # model_file = dataRoot + 'model.sav'
            # pickle.dump(model, open(model_file, 'wb'))


    app.run_server(debug=True, port=8053)

if __name__ == '__main__':

    # set parameters
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/pecfin_dynamics/fin_morphodynamics/nucleus_data/20230913_test/"

    # load image data
    visualize_atlas(dataRoot)

