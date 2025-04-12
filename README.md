# 🌿 Optimal Radius of Leaves

This repository contains a mathematical and computational study exploring how leaves curve based on their internal layered structure. The project is based on a **nonlinear multi-layer bending model**, inspired by studies on curled nanotubes, and adapted to biological leaf structures.

---

## 🧪 Overview

Leaves often curve due to differential growth or contraction in their layers. This project explores the **optimal radius of curvature** that a leaf adopts by minimizing the energy functional derived from elasticity theory. The underlying physics is captured through isotropic assumptions for each layer and integrated into a dimensionally-reduced bending energy model.

---

## 📁 Repository Structure

- **`optimal_radius.ipynb`** – The main Jupyter notebook for running simulations, analyzing results, and visualizing the curvature and model behavior.
- **`tubes.py`** – Contains the computational core of the model, including the calculation of energy matrices and the function `optimal_radius`, which computes the predicted curvature from material properties and layer mismatches.
- **`one page theory.pdf`** – A concise, single-page theoretical summary outlining the derivation of the optimal curvature formula, including assumptions, functional definitions, and minimization strategy.

---

## 🧠 Mathematical Model

We compute the optimal curvature `κ*` by minimizing the quadratic form:

$$\kappa^* = \argmin_{\kappa} \overline{Q}_2(\kappa \cdot e_1)$$

$\overline{Q}_2$ can be seen as
the dimensionally reduced linearized energy functional at the identity, which is derived from isotropic elasticity model across two layers, incorporating:

- Layer mismatches (B)

- Thickness and aspect ratios (d, h)

- Lamé parameters derived from Young's modulus and Poisson ratio

The predicted radius is then given by:

$$r = \frac{d}{(h \cdot |\kappa^*|)}$$

Full details are found in one page theory.pdf.

## Applications

- Modeling growth patterns in biological tissues
- Bio-inspired design in soft robotics and flexible materials
- Educational tools for elasticity and geometric mechanics

---

## Acknowledgements

This work stems from a cooperation between me and Prof. Bartels from the Freiburg Mathematics Institute together with Michelle Modert and Tom Masselter from the Biology Institute in Freiburg. This cooperation arose during my Master Thesis.