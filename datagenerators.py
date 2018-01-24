import numpy as np
import pandas as pd


def generate_dataset_0(n_samples=500, set_X=None, show_z=False):
    """
    Generate samples from the CSM:
    Nodes: (X,Y,Z)
    Edges: (Z -> X, Z-> Y, X -> Y)
    
    All variables are binary. 
    
    Designed to generate simpson's paradox.
    
    Args
    ----
    n_samples: int, the number of samples to generate
    
    set_X: array, values to set x

    Returns
    -------
    samples: pandas.DateFrame
    
    """
    p_z = 0.5
    p_x_z = [0.9, 0.1]
    p_y_xz = [0.2, 0.4, 0.6, 0.8]
    
    z = np.random.binomial(n=1, p=p_z, size=n_samples)
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = np.choose(z, p_x_z)
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    p_y = np.choose(x+2*z, p_y_xz)
    y = np.random.binomial(n=1, p=p_y, size=n_samples)
    
    if show_z:
        return pd.DataFrame({"x":x, "y":y, "z":z})
    
    return pd.DataFrame({"x":x, "y":y})


def generate_dataset_1(n_samples=500, set_X=None):
    """
    Generate samples from the CSM:
    Nodes: (X,Y,Z)
    Edges: (Z -> X, Z-> Y, X -> Y)
    
    X is binary, Z and Y are continuous. 
    
    Args
    ----
    n_samples: int, the number of samples to generate
    
    set_X: array, values to set x
                
    Returns
    -------
    samples: pandas.DateFrame
    
    """
   
    z = np.random.uniform(size=n_samples)
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = np.minimum(np.maximum(z,0.1), 0.9)
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 = 2 * z
    y1 = y0 - 0.5

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "z":z})



def generate_dataset_2(n_samples=500, set_X=None):
    """
    Generate samples from the CSM:
    Nodes: (X,Y,Z)
    Edges: (Z -> X, Z-> Y, X -> Y)
    
    X is binary, Z and Y are continuous. 
    
    Args
    ----
    n_samples: int, the number of samples to generate
    
    set_X: array, values to set x
                
    Returns
    -------
    samples: pandas.DateFrame
    
    """
   
    z = np.random.uniform(size=n_samples)
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = np.minimum(np.maximum(z,0.1), 0.8)
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 =  2 * z
    y1 =  np.where(z < 0.2, 3, y0)

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "z":z})


def generate_dataset_3(n_samples=500, set_X=None):
    """
    Generate samples from the CSM:
    Nodes: (X,Y,Z)
    Edges: (Z -> X, Z-> Y, X -> Y)
    
    X is binary, Z and Y are continuous. 
    
    Args
    ----
    n_samples: int, the number of samples to generate
    
    set_X: array, values to set x
                
    Returns
    -------
    samples: pandas.DateFrame
    
    """
   
    z = np.random.uniform(size=n_samples)
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = np.where(z < 0.5, 0, 1)
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 =  np.where(z >= 0.5, -4*(z - 0.5), 0)
    y1 =  np.where(z < 0.5,  -4*(z - 0.5), 0)

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "z":z})

