import ClassificatoreNPS
import numpy as np
import gaussianPuff
import GNN

if __name__ == "__main__":
    print("Pention System initialized.")
    print(f"Available modules: {', '.join([m.__name__ for m in [ClassificatoreNPS, gaussianPuff, GNN]])}")