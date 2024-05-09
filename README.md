# Navier Stokes Solver with Sharp Interface Immersed Boundary Method

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project implements a Navier Stokes solver using a sharp interface immersed boundary method. It is designed to efficiently solve fluid flow problems with complex geometries and moving boundaries. The solver is accelerated using both CPU (C++) and GPU (CUDA) computations, providing high performance and scalability.

## Features

- Solves the Navier Stokes equations in 2D or 3D.
- Implements the sharp interface immersed boundary method for handling complex geometries and moving boundaries.
- Utilizes CUDA acceleration for GPU computing to achieve high performance.
- Supports various boundary conditions and fluid properties.
- Includes visualization tools for analyzing results.

## Installation

To use this solver, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/navier-stokes-solver.git
   cd navier-stokes-solver
