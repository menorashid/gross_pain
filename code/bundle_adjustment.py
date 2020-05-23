from helpers import util, visualize
import urllib
import urllib.request
import bz2
import os
import cv2
import numpy as np
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares

num_cam_params = 15

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * num_cam_params + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(num_cam_params):
        A[2 * i, camera_indices * num_cam_params + s] = 1
        A[2 * i + 1, camera_indices * num_cam_params + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * num_cam_params + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * num_cam_params + point_indices * 3 + s] = 1

    return A

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * num_cam_params)
        for i in range(n_cameras * num_cam_params):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project_old(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def project(points, camera_params, camera_indices):
    # print (points.shape, camera_params.shape)
    im_pts = np.zeros((points.shape[0],2))
    for view in np.unique(camera_indices):
        bin_cam = camera_indices==view
        points_rel = points[bin_cam,:]
        cam = camera_params[view]
        [rvec,tvec] = [vec[:,np.newaxis] for vec in [cam[:3],cam[3:6]]]
        mtx = np.eye(3).ravel()
        mtx[[0,2,4,5]] = cam[6:10]
        mtx = np.reshape(mtx,(3,3))
        dist = cam[10:15]
        dist = dist[np.newaxis,:]
        im_pts_rel, _ = cv2.projectPoints(points_rel[:,np.newaxis,:], rvec, tvec, mtx, dist)
        im_pts[bin_cam,:] = im_pts_rel.squeeze()
    return im_pts

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * num_cam_params].reshape((n_cameras, num_cam_params))
    points_3d = params[n_cameras * num_cam_params:].reshape((n_points, 3))
    # print (camera_params.shape)

    points_proj = project(points_3d[point_indices], camera_params, camera_indices)
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * num_cam_params + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(num_cam_params):
        A[2 * i, camera_indices * num_cam_params + s] = 1
        A[2 * i + 1, camera_indices * num_cam_params + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * num_cam_params + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * num_cam_params + point_indices * 3 + s] = 1

    return A

def original_script():
    BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
    FILE_NAME = "problem-4num_cam_params-7776-pre.txt.bz2"
    URL = BASE_URL + FILE_NAME

    if not os.path.isfile(FILE_NAME):
        urllib.request.urlretrieve(URL, FILE_NAME)

    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = num_cam_params * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    print (f0)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

def main():
    meta_dir = '../data/camera_calibration_frames_try2'
    interval_str = '20200428104445_113700'
    out_dir_dets = os.path.join(meta_dir, 'calib_im_fix_dets')
    out_dir_intrinsic = os.path.join(meta_dir, 'intrinsics')
    out_dir_calib =  os.path.join(meta_dir, 'extrinsics')

    cell_num = 1
    out_file = os.path.join(out_dir_calib, str(cell_num)+'_bundle.npz')
    loaded = np.load(out_file)
    strings = ['points_3d', 'points_2d', 'point_ind', 'camera_params', 'camera_ind']
    [points_3d, points_2d, point_indices, camera_params, camera_indices] = [loaded[key] for key in strings]

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]


    n = num_cam_params * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    print (np.min(f0), np.max(f0), np.mean(f0))

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    x0_new = res['x']
    camera_params_new = np.reshape(x0_new[:camera_params.size],camera_params.shape)
    points_3d_new = np.reshape(x0_new[camera_params.size:],points_3d.shape)
    out_file = out_file[:-4]+'_optimized.npz'
    print (out_file)

    np.savez(out_file,camera_params = camera_params, points_3d = points_3d)

    print (res)
    f0 = fun(res['x'], n_cameras, n_points, camera_indices, point_indices, points_2d)
    print (np.min(f0), np.max(f0), np.mean(f0))

if __name__=='__main__':
    main()
