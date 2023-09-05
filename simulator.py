import numpy as np
from tabulate import tabulate
from IPython.display import display, Latex
import random

def format_complex_to_latex(z):
    """Helper function to format a complex number to a LaTeX-friendly string."""
    real_part = round(z.real, 3)
    imag_part = round(z.imag, 3)
    
    if imag_part == 0:
        return f"{real_part:.3g}"  # 'g' format to remove trailing zeros
    
    if real_part == 0:
        if imag_part == 1:
            return "i"
        if imag_part == -1:
            return "-i"
        return f"{imag_part:.3g}i"

    sign = "+" if imag_part > 0 else "-"
    imag_part = abs(imag_part)
    
    if imag_part == 1:
        return f"{real_part:.3g} {sign} i"
    else:
        return f"{real_part:.3g} {sign} {imag_part:.3g}i"

def display_complex_matrix(matrix):
    """Display a complex matrix (or array) in LaTeX format."""
    matrix = np.array(matrix)
    
    # If the input is a 1-dimensional array, reshape it to a column vector
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(-1, 1)

    latex_rows = []
    for row in matrix:
        latex_row = " & ".join([format_complex_to_latex(x) for x in row])
        latex_rows.append(latex_row)

    latex_matrix = r" \begin{pmatrix} " + r" \\ ".join(latex_rows) + r" \end{pmatrix} "
    
    display(Latex(latex_matrix))

def print_complex_matrix(matrix):
    """Prints a complex matrix in terminal friendly format"""
    # Ensure input is a numpy array
    matrix = np.array(matrix)
    
    # If the matrix is a 1-dimensional array, reshape it to a column vector
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(-1, 1)
    
    # Custom format for complex numbers
    def format_complex(z):
        real_part = round(z.real, 3)
        imag_part = round(z.imag, 3)
        
        # Handle special cases for the imaginary part
        if imag_part == 1:
            imag_str = "i"
        elif imag_part == -1:
            imag_str = "-i"
        else:
            imag_str = f"{imag_part:.3g}i"
        
        # If there's no imaginary part, just return the real part
        if imag_part == 0:
            return f"{real_part:.3g}"  # Using 'g' format to remove trailing zeros
        # If there's no real part, return only the imaginary part
        elif real_part == 0:
            return imag_str
        # Otherwise, return both parts
        else:
            return f"{real_part:.3g} + {imag_str}"

    formatted_matrix = [[format_complex(x) for x in row] for row in matrix]
    
    print(tabulate(formatted_matrix, tablefmt="grid"))

def get_gate_matrix(gate_code):
    """
    Returns the operator matrix associated with gate_code
    Returns null for an invalid gate code
    Valid Gates: x0 x1 y0 y1 z0 z1 h0 h1 cx01 cx10 sw uf0 uf1 uf2 uf3 oracle
    """
    # Define quantum gates as matrices
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    h = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])

    #for use in calculating oracle matrices
    cx_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])# 0 as control, 1 as target
    x0 = np.kron(x, np.eye(2))

    #oracle functions
    uf0 = np.eye(4)
    uf1 = x0.copy() #f1(0) = 1 and f1(1) = 1
    uf2 = cx_01.copy()#f2(0) = 0 and f2(1) = 1
    uf3 = x0.dot(cx_01).dot(x0) #f3(0) = 1 and f3(1) = 0
    oracle = random.choice([uf0, uf1, uf2, uf3])
    gate_dict = {
        "cx01": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),# 0 as control, 1 as target
        "cx10": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]), # 1 as control, 0 as target
        "sw": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        "x0": np.kron(x, np.eye(2)),
        "x1": np.kron(np.eye(2), x),
        "y0": np.kron(y, np.eye(2)),
        "y1": np.kron(np.eye(2), y),
        "z0": np.kron(z, np.eye(2)),
        "z1": np.kron(np.eye(2), z),
        "h0": np.kron(h, np.eye(2)),
        "h1": np.kron(np.eye(2), h),

        "uf0": uf0,
        "uf1": uf1,
        "uf2": uf2,
        "uf3": uf3,
        "oracle": oracle

    }
    
    if gate_code in gate_dict:
        return gate_dict[gate_code]
    else:
        return None

def gate_description_dictionary():
    # Dictionary mapping from gate codes to their descriptions
    gate_descriptions = {
        "x0": "X gate on qubit 0",
        "x1": "X gate on qubit 1",
        "y0": "Y gate on qubit 0",
        "y1": "Y gate on qubit 1",
        "z0": "Z gate on qubit 0",
        "z1": "Z gate on qubit 1",
        "h0": "Hadamard gate on qubit 0",
        "h1": "Hadamard gate on qubit 1",
        "cx01": "CNOT gate with qubit 0 as control and qubit 1 as target",
        "cx10": "CNOT gate with qubit 1 as control and qubit 0 as target",
        "sw": "SWAP gate between qubits 0 and 1",
        "uf0": "Uf matrix for function f0: constant with f(0) = 0 and f(1) = 0",
        "uf1": "Uf matrix for function f1: constant with f(0) = 1 and f(1) = 1",
        "uf2": "Uf matrix for function f2: balanced with f(0) = 0 and f(1) = 1",
        "uf3": "Uf matrix for function f3: balanced with f(0) = 1 and f(1) = 0",
        "oracle": "Random unitary matrix Uf encoding a function f:{0,1}->{0,1}"
    }
    
    return gate_descriptions

"""
This function applies a series of gates to a 2-qubit state vector.
The composed gate matrix, final state-vector as well as all intermediate transformations are returned.
Input:
    gate_matrices -> list of gates to apply in matrix form
    init_vector -> initial state vector of the quantum system
        [By default set to |00>]
Returns:
    state_vectors_prime -> list of transformed state vectors after each operation is applied
        [Includes the initial and final state-vectors]
    composed_gate_matrix -> matrix representing the composition of all gates
    final_state_vector -> the final state-vector after applying all operations
"""
def compose_and_apply_operations(gate_matrices, init_vector = np.array([1, 0, 0, 0], dtype=np.complex128)):

    state_vectors_prime = [init_vector]
    composed_gate_matrix = np.eye(4)
    final_state_vector = init_vector

    for gate in reversed(gate_matrices):
        composed_gate_matrix = composed_gate_matrix.dot(gate)
    
    for gate in gate_matrices:
        final_state_vector = np.dot(gate, final_state_vector)
        state_vectors_prime.append(final_state_vector)
    
    return state_vectors_prime, composed_gate_matrix, final_state_vector

"""
Parses an input sequence of gates and returns a spliced gate sequence and a list of
associated matrices
Input:
    input_sequence -> input sequence of gates delimited by " "
        [I.e "h0 cx01 h0 h1"]]
Returns:
    gate_sequence -> list of gate names
        [I.e ["h0", "cx", "h0", "h1"]]
    gate_matrices -> list of gates in matrix form associated with gate_sequence
"""
def parse_gate_sequence(input_sequence):
    gate_sequence = input_sequence.split(" ")
    gate_matrices = [get_gate_matrix(gate_name) for gate_name in gate_sequence]

    # Finding invalid gate codes
    invalid_gate_codes = {gate_sequence[i] for i, matrix in enumerate(gate_matrices) if matrix is None}

    return gate_sequence, gate_matrices, invalid_gate_codes

"""
Simulates a quantum circuit given the final state vector of a quantum system.
Input:
    state_vector -> the state vector of the quantum system 
    shots -> the number of runs to simulate
Returns:
    results -> a dictionary of results for each possible observed state
        key -> an element of {00, 01, 10, 11}
        value -> number of shots this state was observed
"""
def simulate_circuit(state_vector, shots = 40):
    results = {bin(i)[2:].zfill(2): 0 for i in range(4)}
    for _ in range(shots):
        r = np.random.rand()
        for i in range(4):
            if r < (np.abs(state_vector[:i+1])**2).sum():
                results[bin(i)[2:].zfill(2)] += 1
                break
    return results

def print_results(results):
    print("Results:")
    for k, v in results.items():
        print(f"|{k}>: [{'Q' * v}] {v}")
    print("---Using Tensor Product Convention (Leftmost Qubit is Qubit 0)---")

def print_simulation_steps(gate_sequence, gate_matrices, state_vectors_prime):
    gate_description = gate_description_dictionary()
    for i in range(len(gate_matrices)):
        display_complex_matrix(state_vectors_prime[i])
        print(f"Applying: {gate_description[gate_sequence[i]]}")
        display_complex_matrix(gate_matrices[i])
    
    display_complex_matrix(state_vectors_prime[-1])

def main():
    shots = 100

    gate_dict = gate_description_dictionary()

    welcome_prompt = f"Available Gates: {' '.join(gate_dict.keys())}\n" + \
        "Enter 'exit' to quit the simulator.\n" + \
        "Enter 'help' for a description of the gates"

    input_prompt = "\nEnter a space-delimited gate sequence:\n" +  \
        "==> "

    print(welcome_prompt)
    # User input
    running = True
    while(running):
        try:
            input_sequence = input(input_prompt)
        except EOFError:
            print("exiting...")
            running = False
            continue

        if input_sequence == "exit":
            print("exiting...")
            running = False
            continue
        elif input_sequence == "help":
            print("----------------------------------------------------------")
            for key, value in gate_dict.items():
                print(f"   {key}: {value}")
            print("----------------------------------------------------------")
            continue

        gate_sequence, gate_matrices, invalid_codes = parse_gate_sequence(input_sequence)
        if invalid_codes:
            print(f"The following gate codes are invalid: {' '.join(invalid_codes)}")
            continue
        print("Calculating...")
        state_vectors_prime, composed_gate_matrix, final_state_vector = compose_and_apply_operations(gate_matrices)
        print("Composed Gate Matrix:")
        print_complex_matrix(composed_gate_matrix)
        print("Final State Vector:")
        print_complex_matrix(final_state_vector)
        
        print(f"Simulating circuit for {shots} shots:")
        results=simulate_circuit(final_state_vector, shots=shots)
        print_results(results)

if __name__ == "__main__":
    main()

