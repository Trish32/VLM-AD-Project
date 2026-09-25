"""Trajectory generation, gating and verification.

Everything here consumes a `SceneRepresentation` and produces or judges a
candidate trajectory. Nothing here reads a detector or an occupancy tensor
directly -- that is the point of the scene representation, and keeping these
modules in one package makes a violation of it visible as an import.
"""
