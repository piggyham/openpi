"""Parquet-driven OpenArm pose simulation with a browser frontend.

Input: LeRobot v2 parquet episodes (16-dim ``observation.state`` joint
sequence). Output: a single web page streaming the live MuJoCo simulation
(front / left_wrist / right_wrist cameras) plus all joint angles (numeric
table + scrolling trend charts). See ``README.md`` in this directory.
"""
