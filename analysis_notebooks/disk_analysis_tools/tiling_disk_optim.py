import numpy as np

####################
### Linear model ###
####################
def linear_model(x, y, a, b, z_0):
    return z_0 + a*x + b*y 

def get_lin_cost_fun(x,y,z,dz):
    """returns chi^2 function for linear model"""
    def lin_cost_fun(a,b,z_0): 
        return np.nansum((linear_model(x, y, a, b, z_0) - z)**2 / dz**2)
    return lin_cost_fun

#######################
### Quadratic model ###
#######################
def quadratic_model(x, y, a, b, c, d, e, z_0):
    return z_0 + a*x + b*y + c*x*y + d*x**2 + e*y**2

def get_quad_cost_fun(x,y,z,dz):
    """returns chi^2 function for quadratic model"""
    def quad_cost_fun(a,b,c,d,e,z_0): 
        return np.nansum((quadratic_model(x, y, a, b, c, d, e, z_0) - z)**2 / dz**2)
    return quad_cost_fun


def chi_sq_ndof (minimizer, xvals):  
    """Calculates reduced chi^2 of iminuit minimizer. Costfunction has to be chi^2 cost function"""
    try:
        ndof = len(xvals) - len(minimizer.values)
        chi_sq = minimizer.fval
        return(chi_sq/ndof)
    except:
        raise TypeError #? I dont know which error to raise here :/