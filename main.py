import sympy as sp

def get_constant_c(func_f: sp.Function, x_interval: tuple, y_interval: tuple, expected_value: float = 1) -> float:
    """
    Function for getting the c value, taken from that the formla: ∫∫ c * f(x,y) dx dy = 1, converted to c = 1 / ∫∫ f(x,y) dx dy
    
    Parameters
    ----------
    func_f: sp.Function 
        the function f(x,y) to be integrated
    x_interval: tuple
        the interval for x, given as a tuple (x_min, x_max)
    y_interval: tuple
        the interval for y, given as a tuple (y_min, y_max)
    expected_value: float
        the expected value of the integral (default is 1)

    Returns
    -------
    float
        the value of c that satisfies the equation
    """
    # calculate the integral and then return the c value
    double_integral = sp.integrate(func_f, 
                                   (y, y_interval[0], y_interval[1]), 
                                   (x, x_interval[0], x_interval[1]))       # result of this is function with 1 parameter c, so we can solve for c
    c_equation = sp.Eq(double_integral, expected_value)  # create eqasion (result of the dobule integral = expected_value (usualy 1))
    c_solution = sp.solve(c_equation, c) # solve for c and if its legit then return it
    return c_solution[0] if c_solution is not None else None

def get_marginal_distribution(func: sp.Function, interval: tuple, variable: sp.Symbol) -> sp.Function:
    marginal_distribution = sp.integrate(func, (variable, interval[0], interval[1]))
    return marginal_distribution

if __name__ == "__main__":

    x, y, c = sp.symbols('x y c')
    fnc: sp.Function = c * x * y                   # function to be investigated

    x_interval = (1, 2)              # intervals for random variables
    y_interval = (x, 2)

    c_value = get_constant_c(fnc, x_interval, y_interval)
    print("Calculating the constant c for the function f(x,y) = c * x * y over the intervals x in [1, 2] and y in [x, 2]...")
    print(f"The constant c is: {c_value}")

    fnc = fnc.subs(c, c_value)  # update the function with newlz get c value substitute the value of c into the function

    marginal_x = get_marginal_distribution(fnc, x_interval, x)
    marginal_y = get_marginal_distribution(fnc, y_interval, y)
    print(f"The marginal distribution for x is: {marginal_x}")
    print(f"The marginal distribution for y is: {marginal_y}")