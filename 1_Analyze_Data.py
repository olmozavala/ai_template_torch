# Common
import sys
sys.path.append("eoas_pyutils")
sys.path.append("ai_common")
# External
import xarray as xr
import os
from os.path import join
import numpy as np
import h5py
# Local
from viz_utils.eoa_viz import EOAImageVisualizer
from proc_utils.geometries import intersect_polygon_grid
from shapely.geometry import Polygon

#%%  Reading a single example
print("Reading data...")
ssh_file = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/SingleExample/dataset-duacs-rep-global-merged-allsat-phy-l4_2010.nc"
ds = xr.open_dataset(ssh_file)
lats = ds.latitude.values
lons = ds.longitude.values

print(f"min lat: {np.amin(lats)}, max lat: {np.amax(lats)}")
print(f"min lon: {np.amin(lons)}, max lon: {np.amax(lons)}")
if np.amax(lons) > 180:
    lons = lons - 360
print("Done!")

#%%
# Plotting some SSH data
print(ds.head())
viz_obj = EOAImageVisualizer(lats=lats, lons=lons)
# viz_obj.plot_3d_data_npdict(ds,  var_names=['adt'], z_levels=[0,1,2], title='SSH', file_name_prefix='')

#%% Reading some contour data
contours_file = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/GOFFISH_UGOS3/ProgramsRepo/data/Geodesic/raw/eddies_altimetry_2010_7days_4.mat"
contours = h5py.File(contours_file, 'r')
#%%
num_contours = len([x for x in contours.keys() if x.startswith('xb0')])
contours_mask = np.zeros_like(ds['adt'][0, :, :])

#%%
all_contours_polygons = []
for i in range(1, num_contours+1):
    cont_lons = contours[f'xb0_{i}'][0, :] - 360
    cont_lats = contours[f'yb0_{i}'][0, :]

    geom_poly = [(cont_lons[i], cont_lats[i]) for i in range(len(cont_lons))]
    all_contours_polygons.append(Polygon(geom_poly))

    intersect_polygon_grid(contours_mask, lats, lons, geom_poly, 1).data

viz_obj.__setattr__('additional_polygons', all_contours_polygons)
viz_obj.plot_3d_data_npdict(ds, var_names=['adt'], z_levels=[0], title='SSH', file_name_prefix='')
print("Done!")
viz_obj.plot_2d_data_np(contours_mask, ['binary_grid'], flip_data=False, rot_90=False, title=F'Intersection Example', file_name_prefix='Test')