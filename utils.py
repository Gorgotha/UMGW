import numpy as np
import ot
import ot.plot
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.datasets import fetch_openml
from scipy.spatial import cKDTree #for voronoi cell subsampling
from scipy.sparse import coo_matrix

from skimage import measure
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_samples_mat(xs, xt, G, thr=1e-8, figsize=(7, 7), plot_X = False, plot_centers = False, **kwargs):
    """ 
    <Altered version of the original implementation taken from: https://pythonot.github.io>

    Plot matrix M  in 2D with  lines using alpha values

    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples


    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        parameters given to the plot functions (default color is black if
        nothing given)
    """

    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    plt.figure(figsize=figsize)
    if plot_X:
        plt.scatter(xs[:, 0], xs[:, 1], s = 1, label='Source samples')
    if plot_centers:
        centers = G.dot(xt)
        print(centers)
        plt.scatter(centers[:, 0], centers[:, 1], s = 1, label='Centers')
    if plot_X or plot_centers:
        plt.legend(loc=0)
        plt.title('Source and target distributions')
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                         alpha=G[i, j] / mx, **kwargs)


def atomic2img(pos, heights, n_pix, extent=None, mode='gaussian', sigma=0.4):
    '''
    Author: Johannes von Lindheim <https://github.com/jvlindheim>
    Creates a pixel image from a discrete measure by interpolation/rounding.

    extent=None: [[x_min, x_max], [y_min, y_max]], specifies where the discrete measure points are placed within the image.
    If None is given, this is automatically generated from the input positions.
    n_pix: desired dimensions (resolution) of the image.

    If mode == 'gaussian', the 'sigma' keyword has to be specified.
    '''
    if extent is None:
        extent = pos2extent(pos)
    assert len(n_pix) == 2, "n_pix has to have length 2 (deal with 2d images)"
    assert extent.shape == (2, 2), "extent has to have shape (2, 2)"
    result = np.zeros(n_pix)
    n_pix = np.array(n_pix)

    if mode == 'gaussian':
        x, y = np.linspace(extent[0, 0], extent[0, 1], n_pix[0]), np.linspace(extent[1, 0], extent[1, 1], n_pix[1])
        xx, yy = np.meshgrid(x, y, indexing='xy')
        yy = yy[::-1]
        pixdists = (extent[:, 1]-extent[:, 0])/np.array(n_pix)
        return (heights[:, None, None]*np.exp(-0.5*((xx[None]-pos[:, 0, None, None])**2/(sigma*pixdists[0])**2 + (yy[None]-pos[:, 1, None, None])**2/(sigma*pixdists[1])**2))).sum(axis=0)
    elif mode == 'bilinear':
        pos = (pos - extent[:, 0][None, :]) / (extent[:, 1] - extent[:, 0])[None, :] * np.array(n_pix)[None, :]
        for i in range(2):
            pos[:, i] = np.clip(pos[:, i], a_min=0.5, a_max=n_pix[i]-0.5-1e-8)
        rounded = np.floor(pos - 0.5).astype(int)
        for i in range(2):
            for j in range(2):
                result[rounded[:, 0]+i, rounded[:, 1]+j] += heights * (1-np.abs(pos[:, 0]-0.5+i - (np.floor(pos[:, 0]-0.5+i)+i))) \
                                                            * (1-np.abs(pos[:, 1]-0.5+j - (np.floor(pos[:, 1]-0.5+j)+j)))
        return result[:, ::-1].T
    else:
        raise ValueError(f"unknown mode {mode}")


def mm_space_from_img(img,metric="euclidean",normalize_meas =True,fused=False):
    assert img.ndim == 2, "img needs to be 2d array"
    if fused:
        supp = np.dstack(np.where(img != 0))[0]
        tmp = img[supp[:,0],supp[:,1]]
        labels = np.sign(tmp)
        height = np.abs(tmp)
        if normalize_meas:
            height/= np.sum(height)
        M = ot.dist(supp,supp,metric=metric)
        return supp,M,height,labels
    else:
        supp = np.dstack(np.where(img > 0))[0]
        height = img[supp[:,0],supp[:,1]]
        if normalize_meas:
            height /= np.sum(height)
        M = ot.dist(supp,supp,metric=metric)
        return supp,M,height