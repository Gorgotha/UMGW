import numpy as np
import warnings
from scipy.special import kl_div
import ot
import ot.plot

import matplotlib.pyplot as plt
import imageio
import os
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.datasets import fetch_openml
from scipy.spatial import cKDTree #for voronoi cell subsampling
from scipy.sparse import coo_matrix

from skimage import measure
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
from copy import deepcopy
from tqdm.notebook import trange
import cv2
import time
import anytree
from sklearn.manifold import MDS
from tqdm import tqdm
from scipy.spatial.distance import squareform, pdist, cdist

def tree_sinkhorn_torch(root, eps, rho,
    divergence='KL',
    max_iter=10000,
    cvgce_thres=1e-5, 
    n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False):
    '''
    Author: Johannes von Lindheim <https://github.com/jvlindheim>
    
    Unbalanced Sinkhorn algorithm, which computes the multimarginal transport between
    histograms with tree-structured costs on box domains. The data is given in form
    of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See Haasler, Isabel, et al. "Multi-marginal Optimal Transport and Schr\" odinger Bridges on Trees." arXiv preprint arXiv:2004.06909 (2020).

    root: root node of the tree. Every node needs to have certain properties, e.g.:
        cost: Backward directed OT costs, i.e. cost has shape (parent_shape, child_shape), then
        multiplication with the corresponding K (e.g. Kv) leads from the child
        to the parent, so that u_iK_ijv_j = pi_ij, i.e. number of rows matches parent.
        (possibly) mu: histogram (masked array). If not given, barycenter is computed for this node.
        However, root node needs to have the mu parameter and will dictate the shape of the domain,
        which all measures need to have. (This restriction is really only for being able to 
        use the Lebesgue measure as reference measure for the unknowns and readily know its shape.)
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. 
        If np.inf is given, compute balanced regularized transport. 
        If None is given, individual rhos are used for every measure 
        (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.

    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''

    assert divergence in ['KL', 'TV'], "unknown divergence"

    # init tree
    assert hasattr(root, 'mu'), "root node needs to be a node with given measure mu or domain_shape must be given"
    #lebesgue = np.ones(domain_shape) / np.prod(domain_shape)
    forward = [node for node in anytree.PreOrderIter(root)]
    backward = forward[::-1]
    for node in forward:
        assert hasattr(node, 'cost')
        node.given = (hasattr(node, 'mu'))
        if node.cost is not None:
            node.domain_shape = len(node.cost.T)
        else:
            node.domain_shape = len(node.mu)
        node.u = torch.ones(node.domain_shape)
        node.K = [] if node.is_root else torch.exp(-node.cost/eps)
        node.a_forw = torch.ones(node.domain_shape)
        if not node.is_root:
            node.a_backw = torch.ones(node.parent.domain_shape)
        node.marginal = None
        node.pi_left = None
        node.pi_right = None
        if rho is not None:
            node.rho = torch.Tensor([rho])
        node.exponent = node.rho/(node.rho+eps) if not torch.isinf(node.rho) else 1.0

    # init iterations
    update = np.inf
    it = 0
    memsnapit = 10
    #     tracemalloc.start(1000)
    prevu = None

    # unbalanced multi-sinkhorn iterations
    while (update > cvgce_thres and it < max_iter):
        del prevu
        prevu = deepcopy([node.u for node in forward])

        # forward pass: update scaling variables u and then update alphas in
        # direction from root down towards leaves simultaneously
        for node in forward:
            incoming = torch.stack([child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw]))
            if node.given:
                if node.exponent == 0:
                    node.u = torch.ones(torch.shape(node.u))
                    
                elif divergence == 'KL': #catch error
                    #if torch.any(torch.prod(incoming, dim=0)) == 0:
                    #    print("it:" + str(it))
                    #    print("node:" + str(node.id))
                    #    print("OVERFLOW/UNDERFLOW ERROR")
                    #    return None
                    node.u = ((node.mu / torch.prod(incoming, axis=0))**node.exponent)#*node.itermult
                elif divergence == 'TV':
                    node.u = torch.minimum(torch.exp((rho+0)/eps), torch.maximum(np.exp(-(rho+0)/eps), node.mu/torch.prod(incoming, dim=0)))
            for i, child in enumerate(node.children):
                tmp3 = list(incoming[:i]) + list(incoming[i+1:])
                if len(tmp3) == 0:
                    a_prod = 1
                else:
                    a_prod = torch.prod(torch.stack(tmp3), dim=0)
                child.a_forw = torch.matmul(child.K.T,(node.u*a_prod))

        # backward pass: update alphas in direction from leaves up towards root
        for node in backward[:-1]: # everyone except root
            tmp = [child.a_backw for child in node.children]
            if len(tmp) == 0:
                a_prod = 1
            else:
                a_prod = torch.prod(torch.stack(tmp), dim=0)
            node.a_backw = torch.matmul(node.K,(node.u*a_prod))
            
        it += 1

        # compute updates every couple iterations
        if it % n_its_check_cvgce == 0:
            update = max([torch.abs(node.u - pru).max() \
                          / max(1., (node.u).max(), (pru).max()) \
                        for (node, pru) in zip(forward, prevu)])

            #print(update)
            if verbose >= 2:
                print("-----it {0}, update {1}".format(it, update))
            if np.isinf(update):
                print("Algorithm diverged. Return None.")
                #return None

    # compute marginals and marginal plans
    for node in forward:
        incoming = torch.stack([child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw]))
        node.marginal = node.u * torch.prod(incoming, dim=0)
        if not node.is_root:
            parent_in = ([] if node.parent.is_root else [node.parent.a_forw]) \
                        + [child.a_backw for child in node.parent.children if child != node] # first line: from parent's parent, second line: from parents children except node
            if len(parent_in) == 0:
                tmp_mult = 1
            else:
                tmp_mult = torch.prod(torch.stack(parent_in), dim=0)
            tmp2 = [child.a_backw for child in node.children]
            if len(tmp2) == 0:
                tmp_mult2 = 1
            else:
                tmp_mult2 = torch.prod(torch.stack(tmp2), dim=0)
            node.pi_left = node.parent.u*tmp_mult
            node.pi_right = node.u*tmp_mult2

    #print("Terminated after {0} iterations with last update {1}.".format(it, update))
    return root


def tree_sinkhorn(root, eps, rho,
    divergence='KL',
    max_iter=10000,
    cvgce_thres=1e-5, 
    n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False):
    '''
    Author: Johannes von Lindheim <https://github.com/jvlindheim>

    Unbalanced Sinkhorn algorithm, which computes the multimarginal transport between
    histograms with tree-structured costs on box domains. The data is given in form
    of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See Haasler, Isabel, et al. "Multi-marginal Optimal Transport and Schr\" odinger Bridges on Trees." arXiv preprint arXiv:2004.06909 (2020).

    root: root node of the tree. Every node needs to have certain properties, e.g.:
        cost: Backward directed OT costs, i.e. cost has shape (parent_shape, child_shape), then
        multiplication with the corresponding K (e.g. Kv) leads from the child
        to the parent, so that u_iK_ijv_j = pi_ij, i.e. number of rows matches parent.
        (possibly) mu: histogram (masked array). If not given, barycenter is computed for this node.
        However, root node needs to have the mu parameter and will dictate the shape of the domain,
        which all measures need to have. (This restriction is really only for being able to 
        use the Lebesgue measure as reference measure for the unknowns and readily know its shape.)
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. 
        If np.inf is given, compute balanced regularized transport. 
        If None is given, individual rhos are used for every measure 
        (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.

    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''

    assert divergence in ['KL', 'TV'], "unknown divergence"

    # init tree
    assert hasattr(root, 'mu'), "root node needs to be a node with given measure mu or domain_shape must be given"
    #lebesgue = np.ones(domain_shape) / np.prod(domain_shape)
    forward = [node for node in anytree.PreOrderIter(root)]
    backward = forward[::-1]
    for node in forward:
        assert hasattr(node, 'cost')
        node.given = (hasattr(node, 'mu'))
        if node.cost is not None:
            node.domain_shape = len(node.cost.T)
        else:
            node.domain_shape = len(node.mu)
        node.u = np.ones(node.domain_shape)
        node.K = [] if node.is_root else np.exp(-node.cost/eps)
        node.a_forw = np.ones(node.domain_shape)
        if not node.is_root:
            node.a_backw = np.ones(node.parent.domain_shape)
        node.marginal = None
        node.pi_left = None
        node.pi_right = None
        if rho is not None:
            node.rho = rho
        node.exponent = node.rho/(node.rho+eps) if not np.isinf(node.rho) else 1.0

    # init iterations
    update = np.inf
    it = 0
    memsnapit = 10
    #     tracemalloc.start(1000)
    prevu = None

    # unbalanced multi-sinkhorn iterations
    while (update > cvgce_thres and it < max_iter):
        del prevu
        prevu = deepcopy([node.u for node in forward])

        # forward pass: update scaling variables u and then update alphas in
        # direction from root down towards leaves simultaneously
        for node in forward:
            incoming = [child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw])
            if node.given:
                if node.exponent == 0:
                    node.u = np.ones(np.shape(node.u))
                    
                elif divergence == 'KL': #catch error
                    if np.any(np.prod(incoming, axis=0)) == 0:
                        print("it:" + str(it))
                        print("node:" + str(node.id))
                        print("OVERFLOW/UNDERFLOW ERROR")
                        return None
                    node.u = ((node.mu / np.prod(incoming, axis=0))**node.exponent)#*node.itermult
                elif divergence == 'TV':
                    node.u = np.minimum(np.exp((rho+0)/eps), np.maximum(np.exp(-(rho+0)/eps), node.mu/np.prod(incoming, axis=0)))
            for i, child in enumerate(node.children):
                a_prod = np.prod(incoming[:i] + incoming[i+1:], axis=0)
                child.a_forw = child.K.T.dot(node.u*a_prod)

        # backward pass: update alphas in direction from leaves up towards root
        for node in backward[:-1]: # everyone except root
            a_prod = np.prod([child.a_backw for child in node.children], axis=0)
            node.a_backw = node.K.dot(node.u*a_prod)
            
        it += 1

        # compute updates every couple iterations
        if it % n_its_check_cvgce == 0:
            update = max([np.abs(node.u - pru).max() \
                          / max(1., (node.u).max(), (pru).max()) \
                        for (node, pru) in zip(forward, prevu)])

            #print(update)
            if verbose >= 2:
                print("-----it {0}, update {1}".format(it, update))
            if np.isinf(update):
                print("Algorithm diverged. Return None.")
                #return None

    # compute marginals and marginal plans
    for node in forward:
        incoming = [child.a_backw for child in node.children] + ([] if node.is_root else [node.a_forw])
        node.marginal = node.u * np.prod(incoming, axis=0)
        if not node.is_root:
            parent_in = ([] if node.parent.is_root else [node.parent.a_forw]) \
                        + [child.a_backw for child in node.parent.children if child != node] # first line: from parent's parent, second line: from parents children except node
            node.pi_left = node.parent.u*np.exp(0/eps)*np.prod(parent_in, axis=0)
            node.pi_right = node.u*np.exp(0/eps)*np.prod([child.a_backw for child in node.children], axis=0)

    #print("Terminated after {0} iterations with last update {1}.".format(it, update))
    return root



def compute_local_cost(pi, a, dx, b, dy, eps, rho, rho2, complete_cost=True):
    """
    Author: Thibault Séjourné <https://github.com/thibsej/unbalanced_gromov_wasserstein>
    
    Compute the local cost by averaging the distortion with the current
    transport plan.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    transport plan used to compute local cost

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    complete_cost: bool
    If set to True, computes the full local cost, otherwise it computes the
    cross-part on (X,Y) to reduce computational complexity.

    Returns
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    local cost depending on the current transport plan.
    """
    distxy = torch.einsum(
        "ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi)
    )
    kl_pi = torch.sum(
        pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log()
    )
    if not complete_cost:
        return - 2 * distxy + eps * kl_pi

    mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
    distxx = torch.einsum("ij,j->i", dx ** 2, mu)
    distyy = torch.einsum("kl,l->k", dy ** 2, nu)


    lcost = (distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi

    if rho < float("Inf") and 0 < rho:
        lcost = (
                lcost
                + rho
                * torch.sum(mu * (mu / a + 1e-10).log())
        )
    if rho2 < float("Inf") and 0 < rho2:
        lcost = (
                lcost
                + rho2
                * torch.sum(nu * (nu / b + 1e-10).log())
        )
    return lcost



def UMGW_torch(r, eps, rho,img1 = None,img2 = None,n_its = 80,
    divergence='KL',
    max_iter=10000,plan_cvgce_thres = 1e-6,
    sink_cvgce_thres=1e-5,
    n_its_check_cvgce=10,
    sink_n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False,
    plot_intermediate = False):
    '''
    Computes (Unbalanced) Multi-marginal Gromov--Wasserstein Transport on a tree by alternately
    solving a local UMOT Problem using a Sinkhorn procedure and updating the local costs.
    The data is given in form of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See also: F. Beier, R. Beinert, G. Steid, "Multi-marginal Gromov--Wasserstein Transport and Barycenters" 
    arXiv preprint: https://arxiv.org/abs/2205.06725.

    r: root node of the tree. Every node needs to have certain properties, e.g.:
        M: Internal similarity matrix, e.g. power of metric (np.array).
            Needs to be directed backwards, i.e. M has shape (parent_shape, child_shape)
        (possibly) mu: histogram (np.array). If not given, barycenter is computed for this node.
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. If np.inf is given, compute balanced regularized transport. If None is given, individual rhos are used for every measure (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.
    n_its: maximum number of iterations for the cost/coupling update loop
    
    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''
    
    margs = []
    times = []
    pis = []
    st = time.time()
    
    forward = [node for node in anytree.PreOrderIter(r)]
    update = np.inf
    
    #initialize costs and pis
    for node in forward:
        if node.is_root:
            node.pi = None
            node.cost = None
            continue
        if hasattr(node.parent,"mu"):
            mu1 = node.parent.mu
        else:
            mu1 = torch.from_numpy(ot.unif(len(node.parent.M)))
        if hasattr(node,"mu"):
            mu2 = node.mu
        else:
            mu2 = torch.from_numpy(ot.unif(len(node.M)))
        node.pi = mu1[:,None] * mu2[None,:]
        node.pi /= torch.sum(node.pi)
        node.cost = compute_local_cost(node.pi, mu1, node.parent.M, mu2, node.M, eps, node.parent.rho, node.rho)

    #main loop
    for i in trange(n_its):
        #rescale eps,rho according to total mass (doesnt do anything in the balanced case)
        tmp = torch.sum(forward[1].pi)
        epstilde = eps * tmp
        for node in forward:
            node.rho = node.rho * tmp
            
        #Sinkhorn for the local problem
        epstilde = 1e-3 * max([torch.max(node.cost) for node in forward[1:]])
        r = tree_sinkhorn_torch(r,epstilde,rho,divergence = divergence,max_iter=max_iter,cvgce_thres=sink_cvgce_thres,n_its_check_cvgce=n_its_check_cvgce,pot_large=pot_large,verbose=verbose)
        times.append(time.time() - st)
        margs.append(n1.marginal)        
        #undo rho rescaling
        for node in forward:
            node.rho = node.rho / tmp
        
        #Update pis and costs
        for node in forward:
            if node.is_root:
                continue
                
            #rescale pi (doesnt do anything for the balanced case)
            old_pi = node.pi
            node.old_pi = old_pi
            new_pi = node.pi_left[:, None] * node.K * node.pi_right[None, :] #/ node.marginal
            tmp2 = torch.sqrt(torch.sum(new_pi)/torch.sum(old_pi))
            node.pi =  (1/tmp2) * new_pi

            #update costs
            node.cost = compute_local_cost(node.pi, torch.from_numpy(ot.unif(len(node.parent.M))), node.parent.M, torch.from_numpy(ot.unif(len(node.M))), node.M, eps, node.parent.rho, node.rho)
            
        pis.append([node.pi for node in forward[1:]])
        if i % n_its_check_cvgce == 0:
            update = max([torch.abs(node.old_pi - node.pi).max() \
                          / max(1., node.old_pi.max(), node.pi.max()) \
                        for node in forward[1:]])
        i += 1
        if update < plan_cvgce_thres:
            break
    
    print("Terminated after {0} iterations with last update {1}.".format(i, update))
    return margs,pis,times
    return r




def UMGW(r, eps, rho,img1 = None,img2 = None,n_its = 80,
    divergence='KL',
    max_iter=10000,plan_cvgce_thres = 1e-6,
    sink_cvgce_thres=1e-5,
    n_its_check_cvgce=10,
    sink_n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False,
    plot_intermediate = False):
    '''
    Computes (Unbalanced) Multi-marginal Gromov--Wasserstein Transport on a tree by alternately
    solving a local UMOT Problem using a Sinkhorn procedure and updating the local costs.
    The data is given in form of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See also: F. Beier, R. Beinert, G. Steid, "Multi-marginal Gromov--Wasserstein Transport and Barycenters" 
    arXiv preprint: https://arxiv.org/abs/2205.06725.

    r: root node of the tree. Every node needs to have certain properties, e.g.:
        M: Internal similarity matrix, e.g. power of metric (np.array).
            Needs to be directed backwards, i.e. M has shape (parent_shape, child_shape)
        (possibly) mu: histogram (np.array). If not given, barycenter is computed for this node.
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. If np.inf is given, compute balanced regularized transport. If None is given, individual rhos are used for every measure (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.
    n_its: maximum number of iterations for the cost/coupling update loop
    
    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''
    
    
    forward = [node for node in anytree.PreOrderIter(r)]
    update = np.inf
    
    #initialize costs and pis
    for node in forward:
        if node.is_root:
            node.pi = None
            node.cost = None
            continue
        if hasattr(node.parent,"mu"):
            mu1 = node.parent.mu
        else:
            mu1 = ot.unif(len(node.parent.M))
        if hasattr(node,"mu"):
            mu2 = node.mu
        else:
            mu2 = ot.unif(len(node.M))
        node.pi = mu1[:,None] * mu2[None,:]
        node.pi /= np.sum(node.pi)
        node.cost = np.array(compute_local_cost(torch.from_numpy(node.pi), torch.from_numpy(mu1), torch.from_numpy(node.parent.M), torch.from_numpy(mu2), torch.from_numpy(node.M), eps, node.parent.rho, node.rho))
    
    #main loop
    for i in trange(n_its):
        #rescale eps,rho according to total mass (doesnt do anything in the balanced case)
        tmp = np.sum(forward[1].pi)
        epstilde = eps * tmp
        for node in forward:
            node.rho = node.rho * tmp
            
        #Sinkhorn for the local problem
        r = tree_sinkhorn(r,epstilde,rho,divergence = divergence,max_iter=max_iter,cvgce_thres=sink_cvgce_thres,n_its_check_cvgce=n_its_check_cvgce,pot_large=pot_large,verbose=verbose)
        
        #undo rho rescaling
        for node in forward:
            node.rho = node.rho / tmp
        
        #Update pis and costs
        for node in forward:
            if node.is_root:
                continue
                
            #rescale pi (doesnt do anything for the balanced case)
            old_pi = node.pi
            node.old_pi = old_pi
            new_pi = node.pi_left[:, None] * node.K * node.pi_right[None, :] #/ node.marginal
            tmp2 = np.sqrt(np.sum(new_pi)/np.sum(old_pi))
            node.pi =  (1/tmp2) * new_pi

            #update costs
            node.cost = np.array(compute_local_cost(torch.from_numpy(node.pi), torch.from_numpy(ot.unif(len(node.parent.M))), torch.from_numpy(node.parent.M), torch.from_numpy(ot.unif(len(node.M))), torch.from_numpy(node.M), eps, node.parent.rho, node.rho))
        if i % n_its_check_cvgce == 0:
            update = max([np.abs(node.old_pi - node.pi).max() \
                          / max(1., node.old_pi.max(), node.pi.max()) \
                        for node in forward[1:]])
        i += 1
        if update < plan_cvgce_thres:
            break
    
    print("Terminated after {0} iterations with last update {1}.".format(i, update))
    return r





def compute_local_cost_sep(pi, a, dx, b, dy, eps, rho, rho2,dx_sep = None,dy_sep=None, complete_cost=True):
    """
    Author: Thibault Séjourné
    https://github.com/thibsej/unbalanced_gromov_wasserstein
    
    Compute the local cost by averaging the distortion with the current
    transport plan.

    Parameters
    ----------
    pi: torch.Tensor of size [Batch, size_X, size_Y]
    transport plan used to compute local cost

    a: torch.Tensor of size [Batch, size_X]
    Input measure of the first mm-space.

    dx: torch.Tensor of size [Batch, size_X, size_X]
    Input metric of the first mm-space.

    b: torch.Tensor of size [Batch, size_Y]
    Input measure of the second mm-space.

    dy: torch.Tensor of size [Batch, size_Y, size_Y]
    Input metric of the second mm-space.

    eps: float
    Strength of entropic regularization.

    rho: float
    Strength of penalty on the first marginal of pi.

    rho2: float
    Strength of penalty on the first marginal of pi. If set to None it is
    equal to rho.

    complete_cost: bool
    If set to True, computes the full local cost, otherwise it computes the
    cross-part on (X,Y) to reduce computational complexity.

    Returns
    ----------
    lcost: torch.Tensor of size [Batch, size_X, size_Y]
    local cost depending on the current transport plan.
    """

    if dx_sep is None and dy_sep is None:
        distxy = torch.einsum(
            "ij,kj->ik", dx, torch.einsum("kl,jl->kj", dy, pi)
        )
    elif dy_sep is not None:
        l = len(dy_sep)
        distxy1 = torch.einsum(
        "ij,kj->ik", dx,torch.einsum("kl,jl->kj", dy_sep, torch.sum(pi.reshape(-1,l,l),dim=1))) 
        distxy2 = torch.einsum(
        "ij,kj->ik", dx,torch.einsum("kl,jl->kj", dy_sep, torch.sum(pi.reshape(-1,l,l),dim=2)))
        distxy = torch.kron(distxy2,torch.ones(l)) + torch.kron(torch.ones(l),distxy1)
        
    elif dx_sep is not None:
        l = len(dx_sep)
        distxy1 = torch.einsum(
        "ij,kj->ik", dx_sep, torch.einsum("kl,jl->kj", dy, torch.sum(pi.reshape((l,l,-1)),dim=0)))
        distxy2 = torch.einsum(
        "ij,kj->ik", dx_sep, torch.einsum("kl,jl->kj", dy, torch.sum(pi.reshape((l,l,-1)),dim=1)))
        distxy = (torch.kron(torch.ones(l,1),distxy1) + torch.kron(distxy2,torch.ones(l,1)))

    kl_pi = torch.sum(
        pi * (pi / (a[:, None] * b[None, :]) + 1e-10).log()
    )

    if not complete_cost:
        return - 2 * distxy + eps * kl_pi

    mu, nu = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
    distxx = torch.einsum("ij,j->i", dx ** 2, mu)
    distyy = torch.einsum("kl,l->k", dy ** 2, nu)
    
    lcost = (distxx[:, None] + distyy[None, :] - 2 * distxy) + eps * kl_pi

    if rho < float("Inf") and 0 < rho:
        lcost = (
                lcost
                + rho
                * torch.sum(mu * (mu / a + 1e-10).log())
        )
    if rho2 < float("Inf") and 0 < rho2:
        lcost = (
                lcost
                + rho2
                * torch.sum(nu * (nu / b + 1e-10).log())
        )
    return lcost






def UMGW_sep(r, eps, rho
             ,n_its = 80,
    divergence='KL',
    max_iter=10000,plan_cvgce_thres = 1e-6,
    sink_cvgce_thres=1e-5,
    n_its_check_cvgce=10,
    sink_n_its_check_cvgce=10,
    pot_large=1e10,
    verbose=False,log=False,max_time = np.inf,epsmode="normal",random_state = None):
    '''
    Computes (Unbalanced) Multi-marginal Gromov--Wasserstein Transport on a tree by alternately
    solving a local UMOT Problem using a Sinkhorn procedure and updating the local costs.
    The data is given in form of a root node with certain properties (see below).
    As divergences, use Kullback-Leibler.

    See also: F. Beier, R. Beinert, G. Steid, "Multi-marginal Gromov--Wasserstein Transport and Barycenters" 
    arXiv preprint: https://arxiv.org/abs/2205.06725.

    r: root node of the tree. Every node needs to have certain properties, e.g.:
        M: Internal similarity matrix, e.g. power of metric (np.array).
            Needs to be directed backwards, i.e. M has shape (parent_shape, child_shape)
        (possibly) mu: histogram (np.array). If not given, barycenter is computed for this node.
    eps: sinkhorn regularization parameter
    rho: unbalanced regularization parameter. If np.inf is given, compute balanced regularized transport. If None is given, individual rhos are used for every measure (useful e.g. for computing barycenters with custom weights), which are saved in the nodes.
    n_its: maximum number of iterations for the cost/coupling update loop
    
    returns:
    root node of tree where now each marginal and marginal plan is computed.
    '''
    
    if random_state is not None:
        np.random.seed(random_state)
    margs = []
    if log:
        times = []
        plans = []
    st = time.time()
    
    forward = [node for node in anytree.PreOrderIter(r)]
    update = np.inf
    
    #initialize costs and pis
    for node in forward:
        if node.is_root:
            node.pi = None
            node.cost = None
            continue
        else:
            if not hasattr(node,"t"):
                node.t = 1

        if hasattr(node.parent,"mu"):
            mu1 = node.parent.mu
        else:
            mu1 = torch.from_numpy(ot.unif(len(node.parent.M)))
        if hasattr(node,"mu"):
            mu2 = node.mu
        else:
            mu2 = torch.from_numpy(ot.unif(len(node.M)))
           
        node.pi = mu1[:,None] * mu2[None,:]
        node.pi /= torch.sum(node.pi)
        node.cost = node.t * compute_local_cost_sep(node.pi, mu1, node.parent.M, mu2, node.M, eps, node.parent.rho, node.rho,dx_sep = node.parent.sep,dy_sep = node.sep)
    
    #main loop
    i = 0
    update = np.inf
    #epstilde = eps * max([torch.max(node.cost) for node in forward[1:]])
    while(i < n_its and update > plan_cvgce_thres and time.time() - st < max_time):
        #rescale eps,rho according to total mass (doesnt do anything in the balanced case)
        tmp = torch.sum(forward[1].pi)

        for node in forward:
            node.rho *= tmp
        r = tree_sinkhorn_torch(r,tmp * eps,rho,divergence = divergence,max_iter=max_iter,cvgce_thres=sink_cvgce_thres,n_its_check_cvgce=n_its_check_cvgce,pot_large=pot_large,verbose=verbose)
        for node in forward:
            node.rho /= tmp
        if log:
            times.append(time.time() - st)

        for node in forward:
            if node.is_root:
                continue
                
            #rescale pi (doesnt do anything for the balanced case)
            old_pi = node.pi
            node.old_pi = old_pi
            new_pi = node.pi_left[:, None] * node.K * node.pi_right[None, :] #/ node.marginal
            tmp2 = torch.sqrt(torch.sum(new_pi)/torch.sum(old_pi))
            node.pi =  (1/tmp2) * new_pi

            #update costs
            node.cost = node.t * compute_local_cost_sep(node.pi, torch.from_numpy(ot.unif(len(node.parent.M))), node.parent.M, torch.from_numpy(ot.unif(len(node.M))),node.M, eps, node.parent.rho, node.rho,dx_sep=node.parent.sep,dy_sep=node.sep)
            #et = time.time()
        if log:
            plans.append([node.pi for node in forward[1:]])
        
        if i % n_its_check_cvgce == 0:
            update = max([np.abs(node.old_pi - node.pi).max() \
                          / max(1., node.old_pi.max(), node.pi.max()) \
                        for node in forward[1:]])
        i += 1

    
    print("Terminated after {0} iterations with last update {1}.".format(i, update))
    if log:
        return r,{"times": times, "plans": plans}
    else:
        return r