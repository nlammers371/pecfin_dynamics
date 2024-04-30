import numpy as np
import pandas as pd
import os
import alphashape
import math
import pyshtools
from vedo import trimesh2vedo, printc, load, spher2cart, mag, ProgressBar, Points, write
from scipy.interpolate import griddata
from tqdm import tqdm
import glob2 as glob
from functions.utilities import path_leaf

def get_SH_coefficients(mesh, N, rmax, x0):

    agrid, pts = [], []
    for th in np.linspace(0, np.pi, N, endpoint=False):
        lats = []
        for ph in np.linspace(0, 2 * np.pi, N, endpoint=False):
            p = spher2cart(rmax, th, ph)
            intersections = mesh.intersect_with_line(x0, x0 + p)
            if len(intersections):
                value = mag(intersections[0] - x0)
                lats.append(value)
                pts.append(intersections[0])
            else:
                lats.append(rmax)
                pts.append(p)
        agrid.append(lats)
    agrid = np.array(agrid)

    grid = pyshtools.SHGrid.from_array(agrid)
    clm = grid.expand()

    return clm


def get_SH_reconstructions(clmCoeffs, lmax, N):
# clmCoeffs = clm #pyshtools.SHCoeffs.from_array(clmSpline[t])

    grid_reco = clmCoeffs.expand(lmax=lmax)

    ##############################################
    agrid_reco = grid_reco.to_array()

    ll = []
    for i, long in enumerate(np.linspace(0, 360, num=agrid_reco.shape[1], endpoint=True)):
        for j, lat in enumerate(np.linspace(90, -90, num=agrid_reco.shape[0], endpoint=True)):
            # th = np.deg2rad(90 - lat)
            # ph = np.deg2rad(long)
            # p = spher2cart(agrid_reco[j][i], th, ph)
            ll.append((lat, long))

    radii = agrid_reco.T.ravel()
    n = N * 1j #2*N * 1j
    lnmin, lnmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
    grid = np.mgrid[lnmax[0]:lnmin[0]:n, lnmin[1]:lnmax[1]:n]
    grid_x, grid_y = grid
    agrid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='linear')

    pts2 = []
    for i, long in enumerate(np.linspace(0, 360, num=agrid_reco_finer.shape[1], endpoint=False)):
        for j, lat in enumerate(np.linspace(90, -90, num=agrid_reco_finer.shape[0], endpoint=True)):
            th = np.deg2rad(90 - lat)
            ph = np.deg2rad(long)
            p = spher2cart(agrid_reco_finer[j][i], th, ph)
            pts2.append(p)

#     mesh2 = Points(pts2, r=20, c="r", alpha=1)
    return np.asarray(pts2)

if __name__ == '__main__':

    # root = "E:\\Nick\\Cole Trapnell's Lab Dropbox\\Nick Lammers\\Nick\\pecfin_dynamics\\fin_morphodynamics\\"
    root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/fin_morphodynamics/"
    date_folder = "20230913"
    file_prefix = "tdTom_prdm_pecfin_40x"
    alpha = 20
    norm_constant = 300

    sh_rmax = 1
    sh_N = 150
    sh_lmax = 32

    # make output directory
    sh_dir = os.path.join(root, "sh_reconstructions", date_folder, "")
    if not os.path.isdir(sh_dir):
        os.makedirs(sh_dir)

    # get list of files to process
    fin_point_dir = os.path.join(root, "fin_interior_points", date_folder, "")
    fin_dataset_list = glob.glob(fin_point_dir + file_prefix + "*")

    for f, file in enumerate(fin_dataset_list):
        # load dataset
        fin_df = pd.read_csv(file, index_col=0)
        fin_points = fin_df.loc[np.where(fin_df["fin_flag"] == 1)[0], ["X", "Y", "Z"]].to_numpy()

        # translate and normalize
        fin_points_norm = (fin_points.copy() - np.min(fin_points, axis=0)) / norm_constant

        # calculate hull for raw data
        raw_fin_hull = alphashape.alphashape(fin_points_norm, alpha)

        # recenter the hull
        raw_centroid = np.mean(np.asarray(raw_fin_hull.vertices), axis=0)
        raw_fin_hull.vertices = raw_fin_hull.vertices - raw_centroid

        # convert to vedo mesh
        vedo_mesh = trimesh2vedo(raw_fin_hull)

        # get SH coefficients
        x0 = np.asarray([0, 0, 0])
        sh_clm = get_SH_coefficients(vedo_mesh, sh_N, sh_rmax, x0)

        # get reconstruction
        sh_reco_pts = get_SH_reconstructions(sh_clm, sh_lmax, sh_N)

        # get alphahull
        sh_fin_hull = alphashape.alphashape(sh_reco_pts, alpha)

        # restor initial dimensions
        raw_fin_hull.vertices = raw_fin_hull.vertices * norm_constant
        sh_fin_hull.vertices = sh_fin_hull.vertices*norm_constant
        sh_reco_pts = sh_reco_pts*norm_constant
        fin_points_norm = fin_points_norm*norm_constant

        # save our results
        filename = path_leaf(file)
        outname = filename.replace(file_prefix + "_", "")
        outname = outname.replace(".csv", "")
        outname = outname.replace("_fin_interior_points", "")
        outdir = os.path.join(sh_dir, outname)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        # save mesh reconstructions
        raw_fin_hull.export(os.path.join(outdir, outname + "_raw_fin_hull.stl"))
        sh_fin_hull.export(os.path.join(outdir, outname + "_sh_reco_fin_hull.stl"))

        # save points
        np.save(os.path.join(outdir, outname + "_raw_fin_points.npy"), fin_points_norm)
        np.save(os.path.join(outdir, outname + "_sh_reco_points.npy"), sh_reco_pts)

        # save sh coefficients
        sh_coeffs = np.asarray(sh_clm.coeffs)
        np.save(os.path.join(outdir, outname + "sh_coeff_array.npy"), sh_coeffs)


    # centroid = np.mean(np.asarray(hull10.vertices), axis=0)
    #
    # SH_dir = os.path.join(figure_path, "SH_reconstructions2")
    # if not os.path.isdir(SH_dir):
    #     os.makedirs(SH_dir)
    #
    # mesh = trimesh2vedo(hull10)
    # centroid = np.mean(np.asarray(hull10.vertices), axis=0)
    # rmax = 1
    # N = 150
    # x0 = centroid
    #
    # lmax = 51
    # l_vec = np.arange(1, lmax)
    #
    # sh_clm_list = []
    # sh_points_list = []
    # sh_hull_list = []
    #
    # for lmax in tqdm([l_vec[-1]]):
    #     sh_clm = get_SH_coefficients(mesh, N, rmax, x0)
    #     sh_points = get_SH_reconstructions(sh_clm, lmax, N)
    #
    #     sh_clm_list.append(sh_clm)
    #     sh_points_list.append(sh_points)
    #
    #     xyz = np.asarray(sh_points)
    #     #     fig = px.scatter_3d(x=xyz[:, 0], y=xyz[:, 1], z=-xyz[:, 2], color=xyz[:, 2], opacity=1)
    #     #     fig.update(layout_coloraxis_showscale=False)
    #     #     fig.update_layout(
    #     #                 scene=dict(aspectratio=dict(x=1, y=1, z=1),
    #     #                     xaxis = dict(nticks=4, range=[-0.3, 0.3],),
    #     #                     yaxis = dict(nticks=4, range=[-0.3, 0.3],),
    #     #                     zaxis = dict(nticks=4, range=[-0.15, 0.15]),))
    #
    #     # #     fig.show()
    #     #     fig.write_image(os.path.join(SH_dir, "sh_fin" + "_lmax" + f"{lmax:03}" + ".png"))
    #
    #     sh_hull = alphashape.alphashape(xyz, 10)
    #
    #     #     tri_points = sh_hull.vertices[sh_hull.faces]
    #     sh_hull_list.append(sh_hull)
