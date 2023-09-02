import numpy as np
from tabulate import tabulate

def print_complex_matrix(matrix):
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


def get_gate_matrix(gate_name):
    # Define quantum gates as matrices
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    h = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    
    if gate_name == "x0":
        gate = np.kron(x, np.eye(2))
    elif gate_name == "x1":
        gate = np.kron(np.eye(2), x)
    elif gate_name == "y0":
        gate = np.kron(y, np.eye(2))
    elif gate_name == "y1":
        gate = np.kron(np.eye(2), y)
    elif gate_name == "z0":
        gate = np.kron(z, np.eye(2))
    elif gate_name == "z1":
        gate = np.kron(np.eye(2), z)
    elif gate_name == "h0":
        gate = np.kron(h, np.eye(2))
    elif gate_name == "h1":
        gate = np.kron(np.eye(2), h)
    elif gate_name == "cx":
        gate = cx
    elif gate_name == "sw":
        gate = sw
    else:
        raise ValueError(f"Unknown gate: {gate_name}")
    
    return gate

"""
This function applies a series of gates to a 2-qubit state vector.
The composed gate matrix, final state-vector as well as all intermediate transformations are returned.
Input:
    gate_matrices -> list of gates to apply in matrix form
    state_vector -> initial state vector of the quantum system
Returns:
    state_vectors_prime -> list of transformed initial state vector after each operation is applied
        [Includes the initial and final state-vectors]
    composed_gate_matrix -> matrix representing the composition of all gates
    final_state_vector -> the final state-vector after applying all operations
"""
def compose_and_apply_operations(gate_matrices, state_vector):

    state_vectors_prime = [state_vector]
    composed_gate_matrix = np.eye(4)
    final_state_vector = state_vector

    for gate in gate_matrices:
        composed_gate_matrix = composed_gate_matrix.dot(gate)
        final_state_vector = np.dot(composed_gate_matrix, state_vector)
        state_vectors_prime.append(final_state_vector)
    
    return state_vectors_prime, composed_gate_matrix, final_state_vector

"""
Parses an input sequence of gates and returns a spliced gate sequence and a list of
associated matrices
Input:
    input_sequence -> input sequence of gates
        [I.e "h0cxh0h1"]]
Returns:
    gate_sequence -> list of gate names
        [I.e ["h0", "cx", "h0", "h1"]]
    gate_matrices -> list of gates in matrix form associated with gate_sequence
"""
def parse_gate_sequence(input_sequence):
    gate_sequence = [input_sequence[i:i+2] for i in range(0, len(input_sequence), 2)]
    gate_matrices = [get_gate_matrix(gate_name) for gate_name in gate_sequence]
    return gate_sequence, gate_matrices

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
            if r < np.abs(state_vector[:i+1]).sum():
                results[bin(i)[2:].zfill(2)] += 1
                break
    return results

def print_results(results):
    print("Results:")
    for k, v in results.items():
        print(f"{k}: [{'Q' * v}] {v}")

def main():
    # Initial state-vector of |00>
    init_vector = np.array([1, 0, 0, 0], dtype=np.complex128)
    shots = 40

    # User input
    input_sequence = input("Enter gate seq (x0,x1,y0,y1,z0,z1,h0,h1,cx,sw): ")

    gate_sequence, gate_matrices = parse_gate_sequence(input_sequence)
    print("Calculating the statevector...")
    state_vectors_prime, composed_gate_matrix, final_state_vector = compose_and_apply_operations(gate_matrices, init_vector)
    print("Composed Gate Matrix:")
    print_complex_matrix(composed_gate_matrix)
    print("Final state vector:")
    print_complex_matrix(final_state_vector)
    
    print(f"Simulating circuit for {shots} shots:")
    results=simulate_circuit(final_state_vector, shots=shots)
    print_results(results)

if __name__ == "__main__":
    main()

