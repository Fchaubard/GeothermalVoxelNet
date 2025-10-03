"""Wrapper script for data preparation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_ode.data_prep import main

if __name__ == "__main__":
    main()
