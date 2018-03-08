import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures



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
        p_x = np.where(z < 0.3, 0, np.where(z > 0.7, 1, 0.7))
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 =  np.where(z >= 0.4, -4*(z - 0.4), 0)
    y1 =  np.where(z < 0.6,  -4*(z - 0.6), 0) + 1

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "z":z})


def generate_exercise_dataset_0(n_samples=500, set_X=None):
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
   
    z = np.random.normal(size=(n_samples, 5))
    beta_treatment = np.array([0,1,2,0,0])
    beta_effect = np.array([1,1,2,0,0])
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = _sigma(np.dot(z, beta_treatment))
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 = np.dot(z, beta_effect)
    y1 = np.dot(z, beta_effect) + 1

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    df = pd.DataFrame({"x":x, "y":y})
    
    for i in range(z.shape[1]):
        df["z_{}".format(i)] = z[:, i]

    return df


def generate_exercise_dataset_1(n_samples=500, set_X=None):
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
   
    z = np.random.normal(size=(n_samples, 5))
    beta_treatment = np.array([-1,-1,-2,0,0])
    beta_effect = np.array([-1,-1,-2,0, 0.5])

    p_x = _sigma(np.dot(z, beta_treatment))
    
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 = np.dot(z, beta_effect)
    y1 = np.dot(z, beta_effect) * (1 + p_x)

    y = np.where(x == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    df = pd.DataFrame({"x":x, "y":y})
    
    for i in range(z.shape[1]):
        df["z_{}".format(i)] = z[:, i]

    return df



def generate_exercise_dataset_2(n_samples=500, set_X=None):
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
    beta_treatment = np.array([ 0.15207176, -0.11653175, -0.34068517,  0.64009405, -0.7243722 ,
       -2.7122607 ,  2.3021001 ,  0.04638091,  1.4096595 , -0.88538833,
       -1.27773486,  1.59597409, -1.27020399,  2.07570976,  0.99324477,
       -0.53702672, -0.10555752,  1.45058372, -1.80245312, -1.92714373,
        1.65904829])
    beta_effect_y0 = np.array([ 0.33313179, -0.04529036,  0.0294476 , -1.57207538, -0.00679557,
        0.87759851, -1.78974391, -0.78558499, -1.50506646, -0.17133791,
        0.7489653 , -0.74583104,  0.79613557, -0.28718545, -1.194678  ,
        0.3952664 , -0.32922775,  0.57037979,  1.19875008,  0.89582566,
       -1.34180865])
    beta_effect_y1 = np.array([-0.8001777 ,  1.16531638, -0.82150055, -0.27853936,  1.74561238,
        0.50031182, -1.74396855, -0.86928906,  0.26423181,  0.01572352,
        1.22709648, -0.08222703, -0.91403023,  0.05014785, -1.34730904,
        0.01790165, -0.60325542,  0.47473682,  0.40199847,  0.49554447,
       -0.13907751])
    
    Z = np.random.normal(size=(n_samples, 5))
    Z2 = PolynomialFeatures().fit_transform(Z)
        
    if set_X is not None:
        assert(len(set_X) == n_samples)
        x = set_X
    else:
        p_x = _sigma(np.dot(Z2, beta_treatment))
        x = np.random.binomial(n=1, p=p_x, size=n_samples)
        
    y0 = np.dot(Z2, beta_effect_y0)
    y1 = np.dot(Z2, beta_effect_y1) + 5
    
    y = np.where(x == 0, y0, y1) + np.random.normal(size=n_samples)
        
    df =  pd.DataFrame({"x":x, "y":y})
    
    for i in range(Z.shape[1]):
        df["z_{}".format(i)] = Z[:, i]
        
    return df



def _sigma(x):
    return 1 / (1 + np.exp(-x))
