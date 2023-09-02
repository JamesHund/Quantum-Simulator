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


def apply_gate(gate_name, statevector):
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
    
    print(f"Applying gate {gate_name}:")
    print_complex_matrix(gate)
    # Apply the gate
    return np.dot(gate, statevector)

def main():
    # Initial statevector
    statevector = np.array([1, 0, 0, 0], dtype=np.complex128)

    # User input
    gate_sequence = input("Enter gate seq (x0,x1,y0,y1,z0,z1,h0,h1,cx,sw): ")

    # Split gate sequence into individual gate names
    gates = [gate_sequence[i:i+2] for i in range(0, len(gate_sequence), 2)]

    print("Calculating the statevector...")
    for gate_name in gates:
        statevector = apply_gate(gate_name, statevector)
        print(".", end="")
    print("Final statevector:")
    print_complex_matrix(statevector)

    shots = 40
    print(f"Running {shots} iterations...")
    results = {bin(i)[2:].zfill(2): 0 for i in range(4)}
    for _ in range(shots):
        r = np.random.rand()
        for i in range(4):
            if r < np.abs(statevector[:i+1]).sum():
                results[bin(i)[2:].zfill(2)] += 1
                break

    print("Results:")
    for k, v in results.items():
        print(f"{k}: [{'Q' * v}] {v}")

if __name__ == "__main__":
    main()

