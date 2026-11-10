
# GSS-Net
A Geometry-Enhanced Serialized State-Space Network for Mountain Tunnel Point Cloud Semantic Segmentation

# Code Availability
This repository currently contains a partial release of the GSS-Net implementation. The complete code will be made publicly available following the publication of the associated research paper.

# Data Preparation
Before using GSS-Net, raw point clouds should be preprocessed into a Pointcept-compatible NumPy format, which may contain the following information:
coord.npy: Point coordinates, with shape (N, 3).
segment.npy: Point-wise semantic labels, with shape (N,).
color.npy: RGB values, with shape (N, 3) (if available).
strength.npy: Laser intensity values (if available).
Here, N denotes the number of points.
Additional geometric features, including local surface normals and PCA-based geometric descriptors, can be generated using the provided feature enhancement tools.