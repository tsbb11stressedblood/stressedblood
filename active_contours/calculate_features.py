import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cell.py as cell

def extract_shape_features(cell):
    # Shape features
    contour = cell.contour
    area = cv2.contourArea(contour)
    nuc_area = sum(sum(cell.nucleus_mask))
    _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
    axis_ratio = major_axis/minor_axis
    perimeter = cv2.arcLength(contour, 1)
    compactness = area/(perimeter*perimeter)
    _,radius = cv2.minEnclosingCircle(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area
    shape_feature_vector = [area, nuc_area, major_axis, minor_axis,axis_ratio, perimeter, compactness, radius, solidity]
    return shape_feature_vector

def extract_statistical_features(cell):
    statistical_feature_vector = []
    return statistical_feature_vector