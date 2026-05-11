"""Project file for MATH5350 Assessment 3"""

# Library Calls

from numba import njit         
import numpy as np
from scipy import linalg
from scipy.stats import norm
import time
from matplotlib import pyplot as plt


def input_data (r, sigma, T, K, M, N, xmin, xmax):
    """Input validation function, returns false when input fails."""

    _fail = False               # fail condition allows for check and inverted result can be passed out

    if not (isinstance (r, (int, float)) and not isinstance (r, bool)) or r <= 0:
        print("The interest rate must be a positive real number,")
        _fail = True

    if not (isinstance (sigma, (int, float)) and not isinstance (r, bool)) or r < 0:
        print("The volatilityinterest rate must be a non-negative real number,")
        _fail = True

    if not (isinstance (T, (int, float)) and not isinstance (T, bool)) or T <= 0:
        print("The maturity date must be a positive real number,")
        _fail = True

    if not (isinstance (K, (int, float)) and not isinstance (K, bool)) or K <= 0:
        print("The strike price must be a positive real number,")
        _fail = True

    if not (isinstance (M, (int, float)) and not isinstance (M, bool)) or M < 0:
        print("The chains must be a non-negative real number,")
        _fail = True

    if not (isinstance (N, (int, float)) and not isinstance (N, bool)) or N < 0:
        print("The sample sizes must be a non-negative real number,")
        _fail = True

    if not (isinstance (xmin, (int, float, np.float64)) and not isinstance (xmin, bool)):
        print("The interest rate must be a number,")
        _fail = True

    if not (isinstance (xmax, (int, float, np.float64)) and not isinstance (xmax, bool)):
        print("The interest rate must be a positive real number,")
        _fail = True
        
    return not _fail


# numba used here to accelerate required for loops
@njit(fastmath=True)
def tridiagonal_solve_thomas (alpha, beta, gamma, b):
    """Thomas algorithm for solving Ax = b with respect to x,

    where A is an n-dimensional square tridiagonal matrix, b is a vector of values length n, and x
    is a vector of values length n.
    
    Expects alpha as the diagonal vector of A, beta as the superdiagonal of A, and gamma as the subdiagonal."""
    
    n = len(b)                  # length of longest diagonal (or dim of matrix)

    # defining our transformed diagonals
    hat_alpha = np.zeros(n, dtype = np.float64) # float64 required for njit efficiency
    hat_b = np.zeros(n, dtype = np.float64)

    # Initial conditions for BTCS loop
    hat_alpha[0] = alpha[0]
    hat_b[0] = b[0]

    # Conversion to simple matrix terms
    for i in range (1, n):
        hat_alpha[i] = alpha[i] - ((beta[i-1] * gamma[i-1]) / hat_alpha[i-1])
        hat_b[i] = b[i] - ((hat_b[i-1] * gamma[i-1]) / hat_alpha[i-1])

    # Solution vector simple calculation
    x = hat_b / hat_alpha
    
    for i in range (n-2, -1, -1):
        x[i] = (hat_b[i] - beta[i] * x[i+1]) / hat_alpha[i]
        
    return x


def thomas_columns (A):
    """Converts a square tridiagonal array into the required Thomas algorithm vectors. Experts a tridiagonal array,
    returns the diagonal, super diagonal, and sub-diagonal as numpy arrays (of numpy.float64s)."""

    n = len(A)
    sub_diag = np.zeros(n-1, dtype = np.float64)
    super_diag = np.zeros(n-1, dtype = np.float64)
    diag = np.zeros(n, dtype = np.float64)

    for i in range (n):         # iterates through central diagonal
        diag[i] = A[i][i]

    for i in range (n-1):
        super_diag[i] = A[i][i+1] # catches value to right of central diagonal

    for i in range (1, n):
        sub_diag[i-1] = A[i][i-1] # catches value to left of central diagonal

    # deep copy is memory inefficient, only use this within other function calls, otherwise create an index to call
    return diag, super_diag, sub_diag


def generate_banded (A, extra_diags): 
    """Utility for generating banded ab matrix for scipy.

    linalg.solve_banded requires a banded matrix of the format ab[u+i-j,j] = A[i,j], where A is the diagonal
    matrix in question (not necessarily tridiagonal). The extra_diags argument is the number of diagonals
    below and above the main diagonal, for a tridiagonal matrix this is simply (1,1).
    Returns the banded matrix as required."""

    n = len(A)
    l, u = extra_diags
    
    banded = np.zeros((l+u+1, n))
    
    for i in range (n):
        for j in range (n):

            if 0 <= u+i-j < l+u+1:
                banded[u+i-j, j] = A[i,j]

    return banded

# We can also compare the speeds of both solvers, accounting for the fact that njit compiles on the first
# function call (taking less time on successive ones), and the banded matrix must be generated for scipy
# to begin with

def generate_tridiagonal_matrix (n, low = 0, high = 10):
    """Utility for sample tridiagonal matrix generation. Expects an integer n, and a tuple range, from low to high,
    and returns a tridiagonal matrix of dimensions n*n. """

    try:
        assert isinstance(n, int)
    except AssertionError:
        print("n must be an integer.")
        return None
    
    mat = np.zeros((n,n), dtype = np.float64)
    # low, high = range

    for i in range (n):
        for j in range (n):

            # central diag
            if i == j:
                mat[i][j] = np.random.uniform(low, high)

            # sub diag
            elif i == j-1:
                mat[i][j] = np.random.uniform(low, high)

            # super diag
            elif i == j+1:
                mat[i][j] = np.random.uniform(low, high)

    return mat

def testing_thomas_accuracy ():
    """Utility for testing whether Thomas algorithm is accurate. Robust against float precision errors."""

    # defining performance testing variables
    N = [100, 1000, 5000, 10000]
    n_trials = 3

    for size in N:

        print(f"Running tests on size {size}...")
        # required formats for both functions
        test_mat = generate_tridiagonal_matrix(size) # actual matrix
        test_alpha, test_beta, test_gamma = thomas_columns(test_mat) # diagonals
        test_banded = generate_banded(test_mat, (1,1)) # scipy banded matrix

        for t in range (n_trials):

            test_b = np.random.uniform(0.0, 10.0, size)    # RHS matrix

            thomas_solutions = tridiagonal_solve_thomas (test_alpha, test_beta, test_gamma, test_b)
            scipy_solutions = linalg.solve_banded((1,1), test_banded, test_b)

            for i in range (t-1):

                try:
                    assert np.isclose(thomas_solutions[i], scipy_solutions[i], rtol = 1e-12, atol = 1e-15)
                except AssertionError:
                    print(f"Error on trial {size}.")
                    return False

    print("All tests successful.")
    return True

def timing_thomas_solvers ():
    """Performance testing for linalg.solve_banded and tridiagonal_solve_thomas.
    
    Requires a tridiagonal matrix A of dimensions n by n, and vector b of length n.
    Returns the required times for both functions to return a vector x of length n such that Ax = b."""

    # numba compilation step call, not tested
    tridiagonal_solve_thomas(np.array([3,4,10]), np.array([6,2]), np.array([9,7]), np.array([10, 6, 2]))

    # defining performance testing variables
    N = [100, 1000, 5000, 10000, 30000]
    thomas_times = []
    scipy_times = []
    n_trials = 5

    for size in N:

        # required formats for both functions
        test_mat = generate_tridiagonal_matrix(size) # actual matrix
        test_alpha, test_beta, test_gamma = thomas_columns(test_mat) # diagonals
        test_banded = generate_banded(test_mat, (1,1)) # scipy banded matrix

        thomas_trial_times = []
        scipy_trial_times = []

        for _ in range (n_trials):

            test_b = np.random.uniform(0.0, 10.0, size)    # RHS matrix

            # Timing tridiagonal Thomas algorithm
            start_time = time.perf_counter_ns()
            tridiagonal_solve_thomas (test_alpha, test_beta, test_gamma, test_b)
            end_time = time.perf_counter_ns()
            
            thomas_trial_times.append(end_time - start_time)

            # Timing scipy algorithm
            start_time = time.perf_counter_ns()
            linalg.solve_banded((1,1), test_banded, test_b)
            end_time = time.perf_counter_ns()
        
            scipy_trial_times.append(end_time - start_time)

        thomas_times.append(np.mean(thomas_trial_times))
        scipy_times.append(np.mean(scipy_trial_times))

    return thomas_times, scipy_times, N


@njit(fastmath=True)
def black_scholes_implicit (r, sigma, T, K, M, N, xmin, xmax):   
    """Implicit calculation for European Call option

    h(S_T) = (S_T-K)^+"""

    # Approximation constants
    delta_tau = ((sigma ** 2) * T) / (2 * N)
    delta_x = (xmax - xmin) / M
    
    llambda = delta_tau / (delta_x ** 2)
    
    # Approximation evolution
    w = np.zeros((M+1,N+1), dtype = np.float64)
    
    # initial period value
    for i in range (M+1):
        x = xmin + i * delta_x
        w[i][0] = max(0., np.exp(x) - K) * np.exp(-r*T)
    
    # central differenced matrix A
    alpha = (1 + 2 * llambda) * np.ones(M-1)
    beta = (-llambda) * np.ones(M-2)
    gamma = (-llambda) * np.ones(M-2)
    
    for v in range (N):

        tau = (v+1) * delta_tau # current time step for discounting
        
        # boundary conditions
        w[0, v+1] = 0.
        w[M, v+1] = np.exp(xmax) - (K * np.exp((-2 * r * tau) / (sigma ** 2)))

        # w_v vector (copy of last chain)
        RHS = w[1:M,v].copy()

        # addition of boundary conditions (RHS is now w_v + d_v)
        RHS[0] += llambda * w[0, v+1]  # left
        RHS[-1] += llambda * w[M, v+1] # right
        
        # Solve for current period step (since Aw_{v+1} = (w_v + d_v))
        w[1:M, v+1] = tridiagonal_solve_thomas(alpha, beta, gamma, RHS)

    return w


def black_scholes_exact (S_0, r, sigma, T, K):

    d_1 = (np.log(S_0 / K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    return S_0 * norm.cdf(d_1) - np.exp(-r * T) * K * norm.cdf(d_2)

def testing_implicit_accuracy (r, sigma, T, K, M, N, xmin, xmax):
    """Implicit calculation for European Call option

    h(S_T) = (S_T-K)^+"""

    # Approximation constants
    delta_tau = ((sigma ** 2) * T) / (2 * N)
    delta_x = (xmax - xmin) / M
    
    llambda = delta_tau / (delta_x ** 2)

    # Approximation evolution
    w = np.zeros((M+1,N+1), dtype = np.float64)
    

    # initial period value
    for i in range (M+1):
        x = xmin + i * delta_x
        w[i][0] = max(0., np.exp(x) - K) # * np.exp(-r * T) 
    

    # central differenced matrix
    alpha = (1 + 2 * llambda) * np.ones(M-1)
    beta = (-llambda) * np.ones(M-2)
    gamma = (-llambda) * np.ones(M-2)
    
    
    for v in range (N):

        tau = (v+1) * delta_tau # current time step for discounting
        
        # boundary conditions        
        w[0, v+1] = 0.
        w[M, v+1] = np.exp(xmax) - (K * np.exp((-2 * r * tau) / (sigma ** 2)))

        # w_v vector (copy of last chain)
        RHS = w[1:M,v].copy()

        # addition of boundary conditions (RHS is now w_v + d_v)
        RHS[0] += llambda * w[0, v+1]  # left
        RHS[-1] += llambda * w[M, v+1] # right
        
        # Solve for current period step (since Aw_{v+1} = (w_v + d_v))
        w[1:M, v+1] = tridiagonal_solve_thomas(alpha, beta, gamma, RHS)

    # exact solutions ========

    x_domain = xmin + np.arange(M+1) * delta_x

    S_domain = np.exp(x_domain) # (x = log(S))

    exact_solution = black_scholes_exact(S_domain, r, sigma, T, K)

    # We can now return the exact solution and the central differenced approximation
    return exact_solution, w, S_domain

@njit(fastmath=True)
def black_scholes_crank_nicolson (r, sigma, T, K, M, N, xmin, xmax):

    # Approximation constants
    delta_tau = ((sigma ** 2) * T) / (2 * N)
    delta_x = (xmax - xmin) / M
    llambda = delta_tau / (delta_x ** 2)

    # Approximation evolution
    w = np.zeros((M+1,N+1), dtype = np.float64)

    # initial period value
    for i in range (M+1):
        x = xmin + i * delta_x
        w[i][0] = max(0., np.exp(x) - K)

    # central differenced matrix
    A_alpha = (2 + 2 * llambda) * np.ones(M-1)
    A_beta = (-llambda) * np.ones(M-2)
    A_gamma = (-llambda) * np.ones(M-2)
        
    for v in range (N):
        
        tau = (v+1) * delta_tau # current time step for discounting
        
        # boundary conditions
        w[0, v+1] = 0.
        w[M, v+1] = np.exp(xmax) - (K * np.exp((-2 * r * tau) / (sigma ** 2)))

        # Crank Nicolson RHS matrix
        RHS = np.zeros(M-1)
        
        for i in range (1,M):
        
            RHS[i-1] = (
                llambda * w[i+1, v] +
                (2 - 2* llambda) * w[i,v] +
                llambda * w[i-1, v]
            )
        
        # addition of boundary conditions (RHS is now w_v + d_v)
        RHS[0] += llambda * (w[0, v+1])  # left
        RHS[-1] += llambda * (w[M, v+1]) # right
        
        # Solve for current period step (since Aw_{v+1} = (w_v + d_v))
        w[1:M, v+1] = tridiagonal_solve_thomas(A_alpha, A_beta, A_gamma, RHS)

    return w 



def testing_rmse (r = 0.05, sigma = 0.1, T = 1.8, K = 673): 

    M_test_cases = [10, 30, 50, 100, 250]
    N_test_cases = [10, 30, 50, 100, 250]
    implicit_rmse = np.zeros ((len(M_test_cases), len(N_test_cases)), dtype = np.float64)
    crank_nicolson_rmse = np.zeros ((len(M_test_cases), len(N_test_cases)), dtype = np.float64)

    n_std = 5
    x_mid = np.log(K) + (r - 0.5 * sigma ** 2)*T
    x_width = n_std * sigma * np.sqrt(T)
    xmin = x_mid - x_width
    xmax = x_mid + x_width

    
    for i, M in enumerate(M_test_cases):      # M discretises X
        for j, N in enumerate(N_test_cases):  # N discretises t
            
            delta_x = (xmax - xmin) / M            
            x_domain = xmin + np.arange(M+1) * delta_x
            S_domain = np.exp(x_domain)
            
            exact = black_scholes_exact(S_domain, r, sigma, T, K)
            implicit = black_scholes_implicit(r, sigma, T, K, M, N, xmin, xmax)
            crank_nicolson = black_scholes_crank_nicolson(r, sigma, T, K, M, N, xmin, xmax)

            if M == 100 and N == 10:
                print(implicit[:,-1])
                print(exact)
            # errors
            implicit_rmse[i,j] = np.sqrt(np.mean((implicit[:,-1] -exact)**2))
            crank_nicolson_rmse[i,j] = np.sqrt(np.mean((crank_nicolson[:,-1] -exact)**2))

    return M_test_cases, N_test_cases, implicit_rmse, crank_nicolson_rmse


            
if __name__ == "__main__":

    
    # This will serve as a simple example case
    example_mat = np.array([
        [3, 6, 0, 0],
        [9, 4, 2, 0],
        [0, 7, 10, 8],
        [0, 0, 12, 4],
    ])

    example_sol = np.array([10, 6, 2, 4])


    tridiagonal_solve_thomas(np.array([3,4,10,4]), np.array([6,2,8]), np.array([9,7,12]), example_sol)
    # This gave me: array([-0.49084249,  1.91208791,  1.38461538, -3.15384615])

    banded_example = generate_banded(example_mat, (1,1))

    # This also gives the solutions array([-0.49084249,  1.91208791,  1.38461538, -3.15384615])
    linalg.solve_banded((1,1), banded_example, example_sol)


    testing_thomas_accuracy()
    

    t_times, s_times, nums = timing_thomas_solvers()
    nums = np.array(nums, dtype = np.float64)
    
    c_linear = t_times[0] / nums[0]
    linear_ref = c_linear * nums

    c_quad = t_times[0] / nums[0]**2
    quad_ref = c_quad * nums**2

    plt.loglog(nums, linear_ref, linestyle = '--', label = r"$O(n)$ reference")
    plt.loglog(nums, quad_ref, linestyle = ':', label = r"$O(n^2)$ reference")

    plt.loglog(nums, t_times, marker = 'o', label = "Thomas algorithm")
    plt.loglog(nums, s_times, marker = 'x', label = "scipy banded algorithm")
    
    plt.title("Comparison of scipy function and Thomas algorithm")
    plt.xlabel("Sample matrix size")
    plt.ylabel("Runtime (logged nanoseconds)")
    plt.legend()
    plt.show()


    

    test_r = 0.05
    test_sigma = 0.1
    test_T = 1.8
    test_K = 673
    test_M = 1000
    test_N = 100
    test_xmin = np.log(test_K/4)
    test_xmax = np.log(test_K*4)

    test_exact, test_num, test_domain = testing_implicit_accuracy (test_r, test_sigma,
                                                                   test_T, test_K, test_M, test_N, test_xmin, test_xmax)


    
    test_num = black_scholes_implicit (test_r, test_sigma,
                            test_T, test_K, test_M, test_N, test_xmin, test_xmax)

    
    abs_error = np.abs(test_num[:,-1] - test_exact)

    max_error = np.max(abs_error)
    
    rmse = np.sqrt(np.mean(abs_error**2))

    test_nic_num = black_scholes_crank_nicolson(test_r, test_sigma,
                                                test_T, test_K, test_M, test_N, test_xmin, test_xmax)

    plt.title("Option Price Estimates")
    plt.xlabel("Time Period")
    plt.ylabel("Price")
    plt.plot(test_domain, test_exact, label="Exact")
    plt.plot(test_domain, test_num[:,0], "--", label="Implicit First Difference (First Run)")
    plt.plot(test_domain, test_nic_num[:,0], "-.", label="Crank Nicolson (First Run)")
    # plt.legend()
    # plt.show()

    # plt.title(f"{test_M}-th estimate for option price")
    # plt.plot(test_domain, test_exact, label="Exact")
    plt.plot(test_domain, test_num[:,-1], "--", label=f"Implicit First Difference {test_M}-th Run")
    plt.plot(test_domain, test_nic_num[:,-1], "-.", label=f"Crank Nicolson {test_M}-th Run")
    plt.legend()
    plt.show()

    print(f"The final exact value is £{test_exact[-1]:.2f}, "
          f"the initial implicit estimate is £{test_num[-1,0]:.2f}, "
          f"and the final implicit estimate is £{test_num[-1,-1]:.2f}")

    print(f"The final exact value is £{test_exact[-1]:.2f}, "
          f"the initial implicit estimate is £{test_nic_num[-1,0]:.2f}, "
          f"and the final implicit estimate is £{test_nic_num[-1,-1]:.2f}")

    
    abs_error_nic = np.abs(test_nic_num[:,-1] - test_exact)

    max_error_nic = np.max(abs_error_nic)
    rmse_nic = np.sqrt(np.mean(abs_error_nic**2))

    print(f"The base model RMSE is £{rmse:.2f}, and the maximum error is £{max_error:.2f}")    
    print(f"The Crank Nicolson RMSE is £{rmse_nic:.2f}, and the maximum error is £{max_error_nic:.2f}")
    rmse - rmse_nic


    M_cases, N_cases, rmse_implicit, rmse_crank = testing_rmse()
    rmse_crank


    im = plt.imshow(
        rmse_implicit,
        origin="lower",
        aspect="auto",
        extent=[
            N_cases[0],
            N_cases[-1],
            M_cases[0],
            M_cases[-1]
        ]
    )

    plt.colorbar(im, label="RMSE")

    plt.xlabel("N (time discretisation)")
    plt.ylabel("M (space discretisation)")

    plt.title("Implicit Scheme RMSE Heatmap")

    plt.show()

    
    im = plt.imshow(
        rmse_crank,
        origin="lower",
        aspect="auto",
        extent=[
            N_cases[0],
            N_cases[-1],
            M_cases[0],
            M_cases[-1]
        ]
    )

    plt.colorbar(im, label="RMSE")

    plt.xlabel("N (time discretisation)")
    plt.ylabel("M (space discretisation)")

    plt.title("Crank Nicolson RMSE Heatmap")

    plt.show()


    rmse_implicit
    rmse_crank
    
