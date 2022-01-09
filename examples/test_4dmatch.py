import numpy as np
from _4dmatch_metric import *
import torch
import time

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
import numpy as np



IR = 0.
NFMR = 0.


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], s=1, color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], s=1, color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')

    plt.draw()
    plt.pause(0.001)


def getitem(entries, index, debug=False):
    entry = np.load(entries[index])

    entry_name = entries[index]
    entry_name = entry_name.split("/")[-2:]
    entry_name = "_".join(entry_name)

    # get transformation
    rot = entry['rot']
    trans = entry['trans']
    s2t_flow = entry['s2t_flow']
    src_pcd = entry['s_pc']
    tgt_pcd = entry['t_pc']
    correspondences = entry['correspondences']  # obtained with search radius 0.015 m
    src_pcd_deformed = src_pcd + s2t_flow
    if "metric_index" in entry:
        metric_index = entry['metric_index'].squeeze()
    else:
        metric_index = np.array([0])

    if (trans.ndim == 1):
        trans = trans[:, None]

    # src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5)
    # tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5)

    rot = rot.astype(np.float32)
    trans = trans.astype(np.float32)
    sflow = (np.matmul(rot, (src_pcd + s2t_flow).T) + trans).T - src_pcd

    src_pcd_raw = src_pcd
    sflow_raw = sflow


    if debug:
        import mayavi.mlab as mlab
        c_red = (224. / 255., 0 / 255., 125 / 255.)
        c_pink = (224. / 255., 75. / 255., 232. / 255.)
        c_blue = (0. / 255., 0. / 255., 255. / 255.)
        scale_factor = 0.013
        mlab.points3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], scale_factor=scale_factor, color=c_red)
        mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
        mlab.quiver3d(src_pcd[:, 0], src_pcd[:, 1], src_pcd[:, 2], sflow[:, 0], sflow[:, 1], sflow[:, 2],
                      scale_factor=1, mode='2ddash', line_width=1.)
        mlab.show()

    return src_pcd, tgt_pcd, sflow, src_pcd_raw, sflow_raw, metric_index, entry_name

def subsample( pc1, pc2, sflow, num_points):
    indice1 = np.arange(pc1.shape[0])
    indice2 = np.arange(pc2.shape[0])
    sampled_indices1 = np.random.choice(indice1, size=num_points, replace=num_points >= pc1.shape[0], p=None)
    sampled_indices2 = np.random.choice(indice2, size=num_points, replace=num_points >= pc2.shape[0], p=None)
    pc1 = pc1[sampled_indices1]
    pc2 = pc2[sampled_indices2]
    sflow = sflow [sampled_indices1]
    return pc1, pc2, sflow



def parallel_func( entries,  i ):
    print(i, "/", len(entries))
    src_pcd_raw, tgt_pcd_raw, sflow_raw, _,_, metric_index , entry_name = \
         getitem(entries, i, debug=False)
    dump_res = os.path.join( res_folder, entry_name )

    src_pcd, tgt_pcd, sflow = subsample(src_pcd_raw, tgt_pcd_raw, sflow_raw, 2048)

    # import open3d as o3d
    # meshA = o3d.geometry.PointCloud()
    # meshA.points = o3d.utility.Vector3dVector(src_pcd)  # [N,3]
    # o3d.visualization.draw_geometries([ meshA])

    st = time.time()

    # fig = plt.figure(figsize=(10,8))
    # ax = fig.add_subplot(111, projection='3d')
    # callback = partial(visualize, ax=ax)

    reg = DeformableRegistration(**{'X': tgt_pcd, 'Y': src_pcd})
    # reg.register(callback)
    reg.register()
    # plt.show()


    tspan = time.time()-st



    gt_flow = torch.from_numpy( sflow[None] )
    pred_flow =torch.from_numpy( reg.TY - src_pcd )[None]
    i_rate = compute_inlier_ratio(pred_flow, gt_flow , inlier_thr=0.04)


    nfmr = compute_nrfmr(
        torch.from_numpy( src_pcd[None]), pred_flow,
        torch.from_numpy(src_pcd_raw[None]),
        torch.from_numpy(sflow_raw[None]), torch.from_numpy(metric_index[None]), recall_thr=0.04)


    a=0

    print(i, "/", len(entries), "IR:", i_rate, "NFMR:",nfmr)
    # print(i, "/", len(dataset), "IR:", IR / (i + 1), "NFMR:", NFMR / (i + 1))
    #
    #
    # np.savez_compressed( dump_res,
    #                      ir=i_rate,
    #                      nfmr=nfmr,
    #                      tspan=tspan)
    #
    #



if __name__ == '__main__':

    import  glob, os
    def read_entries ( split, data_root ):
        # entries = sorted(glob.glob(os.path.join(data_root, split, "*/*.npz")) )
        entries =  glob.glob(os.path.join(data_root, split, "*/*.npz"))
        return entries


    data_root = "/home/liyang/dataset/4DMatch"

    datasplit= {
        "train": "split/train",
        "val": "split/val",
        "test": "split/4DMatch"
    }

    entries = read_entries(  datasplit["test"] , data_root )

    import multiprocessing
    import os
    res_folder = "/home/liyang/workspace/baselines/pyFM/results/LoMatch"


    # pool = multiprocessing.Pool(processes=10)
    for i in range (len(entries)):

        parallel_func (entries,i)
        #
        # pool.apply_async(parallel_func, args=(entries, i))
    #
    # pool.close()
    # pool.join ()

    print("all finish")