from __future__ import annotations

from datetime import datetime
from pathlib import Path
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image
import matplotlib.cm as cm
import pandas as pd
import json
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import convolve
import os
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import scipy.stats
import math

VIDEO_TYPES = [
    ".avi",
    ".mp4",
]
IMAGE_TYPES = [".png", ".bmp", ".tiff", ".jpg", ".jpeg"]


class OCTVolumeWithMetaData(object):
    """Class to hold an OCT volume and any related metadata.

    Also provides methods for viewing and saving.

    Attributes:
        volume: all the volume's b-scans.

        patient_id: patient ID.
        first_name: patient first name.
        surname: patient second name.
        sex: patient sex.
        DOB: patient date of birth.

        volume_id: volume ID.
        acquisition_date: date image acquired.
        num_slices: number of b-scans present in volume.
        laterality: left or right eye.

        contours: contours data.
        pixel_spacing: (x, y, z) pixel spacing in mm.
        metadata: all metadata available in the OCT scan.
    """

    def __init__(
        self,
        volume: list[np.array],
        patient_id: str | None = None,
        first_name: str | None = None,
        surname: str | None = None,
        sex: str | None = None,
        patient_dob: str | None = None,
        volume_id: str | None = None,
        acquisition_date: datetime | None = None,
        laterality: str | None = None,
        contours: dict | None = None,
        pixel_spacing: list[float] | None = None,
        metadata: dict | None = None,
        header: dict | None = None,
        oct_header: dict | None = None,
    ) -> None:
        # image
        self.volume = volume

        # patient data
        self.patient_id = patient_id
        self.first_name = first_name
        self.surname = surname
        self.sex = sex
        self.DOB = patient_dob

        # volume data
        self.volume_id = volume_id
        self.acquisition_date = acquisition_date
        self.laterality = laterality
        self.num_slices = len(self.volume)
        self.contours = contours

        # geom data
        self.pixel_spacing = pixel_spacing

        # metadata
        self.metadata = metadata
        self.header = header
        self.oct_header = oct_header

    def peek(
        self,
        rows: int = 5,
        cols: int = 5,
        filepath: str | Path | None = None,
        show_contours: bool | None = False,
    ) -> None:
        """Plots a montage of the OCT volume. Optionally saves the plot if a filepath is provided.

        Args:
            rows: number of rows in the plot.
            cols: number of columns in the plot.
            filepath: location to save montage to.
            show_contours: if set to ``True``, will plot contours on the OCT volume.
        """
        images = rows * cols
        x_size = rows * self.volume[0].shape[0]
        y_size = cols * self.volume[0].shape[1]
        ratio = y_size / x_size
        slices_indices = np.linspace(0, self.num_slices - 1, images).astype(np.int16)
        plt.figure(figsize=(12 * ratio, 12))
        for i in range(images):
            slice_id = slices_indices[i]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(self.volume[slice_id], cmap="gray")
            if show_contours and self.contours is not None:
                for v in self.contours.values():
                    if (
                        slice_id < len(v)
                        and v[slice_id] is not None
                        and not np.isnan(v[slice_id]).all()
                    ):
                        plt.plot(v[slice_id], color="w")
            plt.axis("off")
            plt.title("{}".format(slice_id))
        plt.suptitle("OCT volume with {} slices.".format(self.num_slices))

        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

    def save(self, filepath: str | Path) -> None:
        """Saves OCT volume as a video or stack of slices.

        Args:
            filepath: location to save volume to. Extension must be in VIDEO_TYPES or IMAGE_TYPES.
        """
        extension = Path(filepath).suffix
        if extension.lower() in VIDEO_TYPES:
            video_writer = imageio.get_writer(filepath, macro_block_size=None)
            for slice in self.volume:
                slice = slice.astype("uint8")
                video_writer.append_data(slice)
            video_writer.close()
        elif extension.lower() in IMAGE_TYPES:
            base = Path(filepath).stem
            print(
                "Saving OCT as sequential slices {}_[1..{}]{}".format(
                    base, len(self.volume), extension
                )
            )
            full_base = Path(filepath).with_suffix("")
            self.volume = np.array(self.volume).astype("float64")
            self.volume *= 255.0 / self.volume.max()
            for index, slice in enumerate(self.volume):
                filename = "{}_{}{}".format(full_base, index, extension)
                cv2.imwrite(filename, slice)
        elif extension.lower() == ".npy":
            np.save(filepath, self.volume)
        else:
            raise NotImplementedError(
                "Saving with file extension {} not supported".format(extension)
            )

    def get_projection(self) -> np.array:
        """Produces a 2D projection image from the volume."""
        projection = np.mean(self.volume, axis=1)
        return projection

    def save_projection(self, filepath: str | Path) -> None:
        """Save a 2D projection image from the volume.

        Args:
            filepath: location to save volume to. Extension must be in IMAGE_TYPES.
        """
        extension = Path(filepath).suffix
        if extension.lower() in IMAGE_TYPES:
            projection = self.get_projection()
            projection = 255 * projection / projection.max()
            cv2.imwrite(filepath, projection.astype(int))
        else:
            raise NotImplementedError(
                "Saving with file extension {} not supported".format(extension)
            )
    
    def derive_scales_and_coord_systems(self):
        # the 'object' (oct_volume) returns width, slice thickness, height - in that order
        # pixel spacing was noted here: https://github.com/marksgraham/OCT-Converter/blob/main/oct_converter/readers/fda.py
        sx = self.pixel_spacing[0]  # mm in terms of width (x axis)
        sz = self.pixel_spacing[1]  # mm in terms of depth (space between spices, z axis)
        sy = self.pixel_spacing[2]  # mm in terms of height / axially (y axis)
    
        # the width & height are per slice, the num_slices (S) is the total slices across whole raster and should be 128 for a UKBB Topcon (here we are in pixel space and not physical space)
        W = self.metadata["img_projection"]['width']
        H = self.metadata["img_projection"]['height']
        S = self.num_slices
        ## Get a coordinate system according to which our macular raster scan will be centred on 0
        x_max = (sx * (W - 1)) /2
        y_max = (sz * (S - 1)) /2
        x_range = np.linspace(-x_max, x_max, W)
        y_range = np.linspace(y_max, -y_max, S)
        X,Y = np.meshgrid(x_range, y_range)
        return sx, sy, sz, W, H, S, x_max, y_max, x_range, y_range, X, Y
    
    def find_fovea(self):# This code follows the principles of Retimat (a matlab repo): https://github.com/mu-biomed/retimat
        sx, sy, sz, W, H, S, x_max, y_max, x_range, y_range, X, Y = self.derive_scales_and_coord_systems()
        keys = set(self.contours.keys())# This is the set of contours saved in the fda file (ie the segmentation coordinates). What is available differs across UK Biobank instances.
    
        ilm_key = "ILM" if "ILM" in keys else ("RETINA_1" if "RETINA_1" in keys else None)# This allows for the inconsistent naming across instances.
        bm_key  = "BM"  if "BM"  in keys else ("RETINA_4" if "RETINA_4" in keys else None)# This allows for the inconsistent naming across instances.
    
        total_thickness_map = np.full((S, W), np.nan, dtype=np.float32)# Create an empty thickness map flled with nan. 
      
        if ilm_key is None or bm_key is None:
            raise ValueError(f"Cannot compute total thickness: missing ILM/BM. Available={sorted(keys)}")# No function I have made works without at least the bm and ilm boundaries present. 
    
        for i in range(self.num_slices):# Iterate thru each slice in the fda file. 
            if not (0 <= i < len(self.volume)):# This is to stop you trying to run on slices that done exist. 
                continue
            ilm = self.contours[ilm_key][i]
            bm  = self.contours[bm_key][i]
            if ilm is None or bm is None:
                # skip this slice instead of killing the whole volume 
                continue
    
            h, w = self.volume[i].shape[:2]
            ilm = np.asarray(ilm, dtype=np.float32).reshape(-1)
            bm  = np.asarray(bm,  dtype=np.float32).reshape(-1)
            if ilm.shape[0] != w or bm.shape[0] != w:
                continue
    
            total = bm - ilm
            total = np.where(np.isfinite(total) & (total >= 0), total, np.nan)
            total_thickness_map[i, :w] = total
    
    
        # convert to µm
        scale = sy * 1000.0# OCT converter explicitly says its scale is in mm in the docs, so multiply by 1000
        total_thickness_map *= scale
        
        # resample - taken from retimat - convert to polar coords
        max_d = 0.85
        n_point=50
        x_grid = np.linspace(-max_d, max_d, n_point)
        y_grid = np.flip(x_grid)
        X1, Y1 = np.meshgrid(x_grid, y_grid)
        is_num = ~np.isnan(total_thickness_map)
        Z1 = griddata(
            points=np.column_stack([X[is_num], Y[is_num]]),
            values=total_thickness_map[is_num],
            xi=(X1, Y1),
            method="cubic"
        )
        Z1 = Z1.reshape(X1.shape)
        kernel_rad = 0.15# this sets the region of interest to 0.15mm radius - a reasonable approximation of foveola radius. You use this to set a neighbourhood of smoothing so you dont find the centre e.g. at just some tiny missegmenged area of hole etc.
        _, rho = cart2pol(X1, Y1)# Now use the helper fx defined previously to get the polar coords
        
        # Now all x,y are instead a certain distance in mm from the origin (0,0) instead of a coord
        kernel_a = rho < kernel_rad
        kernel_b = kernel_a.astype(np.float64).copy()# This is to build a 2D circular mask [0,1] , and T/F above
        kernel_b /= np.sum(kernel_b)
    
        # Now you slide (convolve) your kernel over Z1 (the thickness map) , at each pixel the value is replaed with a average TRT 0.15mm around that area, the 'nearest' mode is about how you deal with it when the kernel extends beyond the image boundary, here we just assume the value would be that of the nearest pixel (i.e. nearest edge)
        TRT_filt = convolve(Z1, kernel_b, mode="nearest")
    
    
        x_fovea, y_fovea = find_min(X1, Y1, TRT_filt)
    
        ## CONVERT BACK TO NATIVE MEASURES (IE SCAN SLICE AND HORIZONTAL PIXEL LOCATION:
        # X and Y were your meshgrid coordinate systems above which define the locations of the raster scan in a cartesian manner
        dist2 = (X - x_fovea)**2 + (Y - y_fovea)**2# You are finding the distance of every pixel from the minimum which is the fovea, you need to square it because negative values are implausible
        i0, j0 = np.unravel_index(np.argmin(dist2), dist2.shape)# argmin finds the lowest index but its not in a coordinate / array style and thats what np.unravel_index is for
        fovea_slice_native = i0   # B-scan index (slice)
        fovea_ascan_native = j0   # A-scan index (where on the width of a scan we need to look) within that B-scan
        
        return fovea_slice_native, fovea_ascan_native
      
    def crop_fovea(self, slice_num=64, fov_loc= 256, suffix=None, output_dir=None, smooth_bm=True):
        if self.contours is None:
            raise ValueError("No contours found on this OCTVolume object.")
        if not (0 <= slice_num < len(self.volume)):
            raise IndexError(f"slice_num={slice_num} out of range 0..{len(self.volume)-1}")
        sx, sy, _, _, _, _, _, _, _, _, _, _ = self.derive_scales_and_coord_systems()# you dont need everything so just asign what you want 
        sigma_px=0.3/sx
        keys = set(self.contours.keys())
        ilm_key = "ILM" if "ILM" in keys else ("RETINA_1" if "RETINA_1" in keys else None)# for newer OCT releases in UKBB you can get other layers but i found instance 2 onwards doesnt work well with this lib because they are wide scans and the optic nerve head creates segmentation problems 
        bm_key  = "BM"  if "BM"  in keys else ("RETINA_4" if "RETINA_4" in keys else None)
        if ilm_key is None or bm_key is None:
            raise ValueError(f"Missing required contours. Have: {sorted(keys)}")
        ilm = self.contours[ilm_key][slice_num]
        bm  = self.contours[bm_key][slice_num]
        h, w = self.volume[slice_num].shape[:2]
        ilm = np.asarray(ilm, dtype=np.float32).reshape(-1)
        bm  = np.asarray(bm,  dtype=np.float32).reshape(-1)
        if ilm.shape[0] != w or bm.shape[0] != w:
            raise ValueError(f"Expected curves of length w={w}, got ilm={ilm.shape}, bm={bm.shape}")
        if smooth_bm:
            x = np.arange(w, dtype=np.float32)
            good_bm = np.isfinite(bm)
            L = int(round(0.5 / sx))
            L = max(L, 5)
            # if L % 2 == 0: L += 1# you must have an odd window length for SG filter << stopped using savgol as it causes odd 'wavy' smooothed segmentations so ive removed this - not relevant to gaussian filter
            bm_y_2 = bm.copy()
            if good_bm.sum() >= 2:
                bm_y_2[~good_bm] = np.interp(x[~good_bm], x[good_bm], bm_y_2[good_bm])
                bm = gaussian_filter1d(bm_y_2, sigma=sigma_px, mode='nearest')
                #bm= savgol_filter(bm_y_2, window_length=L, polyorder=2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0) << removed due to wavy segmentations / forcing a polynomial on straight segments 
                bm[~good_bm] = np.nan
            else:
                # not enough points to fill/smooth
                pass
        mask = _mask_between_ilm_bm(ilm, bm, h=h, w=w)# create a binary of the fovea, uncropped at this stage
        img = self.volume[slice_num]
        crop_region_w = int(round(1.5/sx)) #set rim radius to a ~ 1.5mm based on our fovea paper - https://www.medrxiv.org/content/10.1101/2025.06.27.25330434v1.full-text
        crop_region_h_upper = int(round(0.4/sy))# mm dist converted to pixels
        crop_region_h_lower = int(round(0.4/sy))# mm dist converted to pixels
        rotated, rot_deg, _, _ = rotate_pit_to_bm_vertical(
            img, ilm, bm, fov_loc, sx_mm=sx, sy_mm=sy, window_mm=1)
        mask_rotated, mask_rot_deg, _ , _ = rotate_pit_to_bm_vertical(mask, ilm, bm, fov_loc, sx_mm=sx, sy_mm=sy, window_mm=1)
        img_rotated, img_rot_deg, _, y0 = rotate_pit_to_bm_vertical(img, ilm, bm, fov_loc, sx_mm=sx, sy_mm=sy, window_mm=1)    
        ILM_y = ilm[fov_loc]
        upper_lim = int(round(y0 - crop_region_h_upper))
        lower_lim = int(round(y0 + crop_region_h_lower))
        R_lim = int(round(fov_loc + crop_region_w))
        L_lim = int(round(fov_loc - crop_region_w))
        #print("h,w:", h, w)
        #print("fov_loc:", fov_loc, "ILM_y:", ILM_y)
        #print("y bounds:", upper_lim, lower_lim)
        #print("x bounds:", L_lim, R_lim)
        diff = np.mean(np.abs(rotated.astype(np.float32) - img.astype(np.float32)))
        #print("mean abs diff:", diff)
        img_crop = rotated[upper_lim:lower_lim, L_lim:R_lim]
        mask_crop = mask_rotated[upper_lim:lower_lim, L_lim:R_lim]
        #print("crop shape:", img_crop.shape)
        if suffix is not None:
            #imageio.imwrite(str(save_png), (mask * 255).astype(np.uint8))
            imageio.imwrite(f"{output_dir}/results/foveal_crops/{suffix}fovea.png", img_crop.astype(np.uint8))
            imageio.imwrite(f"{output_dir}/results/central_slices/{suffix}central_slice.png", img.astype(np.uint8))
            imageio.imwrite(f"{output_dir}/results/masked_foveal_crops/{suffix}masked_fovea.png", (mask_crop*255).astype(np.uint8))
        return rot_deg, ILM_y
    
    def plot_thickness_map(
        self,
        angle=0,
        central_slice_num=64,
        fov_loc=None,
        centre_y=None,
        hists=None,
        total_thickness_lower=None,
        total_thickness_upper=None,
        save_dir="./",
        save_prefix="v1_",
        smooth_bm=True
        ):
        """
        Builds a *cropped* perifoveal thickness map (total only here), correcting for
        rigid in-plane scan rotation by rotating ILM/BM polylines and resampling.
    
        Output map shape:
          rows = slices within ±1.5 mm ('slow axis' likely affected by eye movement during imaging which i manage with frequency filtering)
          cols = columns within ±1.5 mm around fov_loc (fast axis)
    
        """
        if fov_loc == None:
            raise ValueError("The x coord for rotation (fov_loc) must be specified")
        
        if centre_y == None:
            raise ValueError("The y coord for rotation (centre_y) must be specified")
      
        t_u16_total = None
    
        angle_radians = np.radians(angle)
    
        # sx: mm per x-pixel (fast axis), sy: mm per y-pixel (axial within slice),
        # sz: mm per slice step (slow axis)
        sx, sy, sz, W, _, H, _, _, _, _, _, _ = self.derive_scales_and_coord_systems()
        sigma_px = 0.3/sx
        
        # determine slice window (±1.5 mm) 
        central_slices_window = int(math.ceil(1.5 / sz))
        lower_slice = int(central_slice_num) - central_slices_window
        upper_slice = int(central_slice_num) + central_slices_window
    
        if (lower_slice < 0) or (upper_slice >= H):
            return np.nan
    
        # determine x crop window (±1.5 mm)
        crop_mm = 1.5
        crop_w_px = int(round(crop_mm / sx))
        c0 = int(fov_loc)
        c1 = max(0, c0 - crop_w_px)
        c2 = min(W, c0 + crop_w_px + 1)
        width_px = int(c2 - c1)
    
        # allocate cropped map: rows are local slice index within [lower_slice, upper_slice]
        height_slices = int(upper_slice - lower_slice + 1)
        total_thickness_map = np.full((height_slices, width_px), np.nan, dtype=np.float32)
    
        keys = set(self.contours.keys())
        ilm_key = "ILM" if "ILM" in keys else ("RETINA_1" if "RETINA_1" in keys else None)
        bm_key  = "BM"  if "BM"  in keys else ("RETINA_4" if "RETINA_4" in keys else None)
    
        if ilm_key is None or bm_key is None:
            raise ValueError(f"Cannot compute total thickness: missing ILM/BM. Available={sorted(keys)}")
    
        # ---- fill map slice-by-slice ----
        for i in range(lower_slice, upper_slice + 1):
            if not (0 <= i < len(self.volume)):
                continue
    
            ilm = self.contours[ilm_key][i]
            bm  = self.contours[bm_key][i]
            if ilm is None or bm is None:
                continue
    
            h, w = self.volume[i].shape[:2]
    
            ilm = np.asarray(ilm, dtype=np.float32)
            bm  = np.asarray(bm,  dtype=np.float32)
            if ilm.shape[0] != w or bm.shape[0] != w:
                continue
    
            # fovea column validity for this slice (if w differs from W, enforce w)
            if not (0 <= c0 < w):
                continue
                
            if smooth_bm:## SG filtering - see refs
                x = np.arange(w, dtype=np.float32)
                good_bm = np.isfinite(bm)
                L = int(round(0.5 / sx))
                L = max(L, 5)
                #if L % 2 == 0: L += 1# you must have an odd window length for SG filter << removed in final version because i switched to gaussian. left here in case chosen to implement later 
                bm_y_2 = bm.copy()
                if good_bm.sum() >= 2:
                    bm_y_2[~good_bm] = np.interp(x[~good_bm], x[good_bm], bm_y_2[good_bm])
                    bm = gaussian_filter1d(bm_y_2, sigma=sigma_px, mode='nearest')
                    #bm= savgol_filter(bm_y_2, window_length=L, polyorder=2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
                    bm[~good_bm] = np.nan
                else:
                    # not enough points to fill/smooth
                    pass
    
            # create origin
            origin_xy = (float(fov_loc), float(centre_y))
    
            # rotate + resample each curve onto x_grid 0..w-1
            ilm_rot = _rotate_curve_and_resample_to_xgrid(ilm, angle_radians, origin_xy)
            bm_rot  = _rotate_curve_and_resample_to_xgrid(bm,  angle_radians, origin_xy)
    
            # thickness in the aligned frame (still per x-grid column)
            total = bm_rot - ilm_rot
            total = np.where(np.isfinite(total) & (total >= 0), total, np.nan)
    
            # crop to [c1:c2] in THIS slice's width
            cc1 = max(0, c1)
            cc2 = min(w, c2)
            row = i-lower_slice
            total_thickness_map[row, :] = total[cc1:cc2]
    
        # convert to µm (assumes sy is axial mm per pixel within the B-scan)
        total_thickness_map *= (sy * 1000.0)
    
        #  plotting extents for the cropped map 
        # x in mm relative to fovea column c0
        x_left_mm  = (c1 - c0) * sx
        x_right_mm = (c2 - c0) * sx
    
        # y in mm relative to central slice (slow axis). origin="upper" => top is smaller row index.
        # Our rows run from i=lower_slice..upper_slice
        y_top_mm    = (lower_slice - central_slice_num) * sz
        y_bottom_mm = (upper_slice - central_slice_num) * sz
    
        crop_extent = [x_left_mm, x_right_mm, y_bottom_mm, y_top_mm]  # bottom..top for origin="upper"
    
        t_norm = (total_thickness_map - total_thickness_lower) / (total_thickness_upper - total_thickness_lower)
        t_norm = np.clip(t_norm, 0, 1)
        t_norm = np.nan_to_num(t_norm, nan=0.0)  # explicit NaN → 0
        t_u16 = (t_norm * 65535).astype(np.uint16)
        t_u16 = cv2.resize(t_u16, (512,512), interpolation=cv2.INTER_CUBIC)
        SM = f'{save_dir}/results/thickness_wavlets/{save_prefix}foveal_thickness_map_wavelet.png'
        features = energy_quantification(t_u16, SM)
        kurtosis = features[1][3]
        if kurtosis > 2.5:
            raise ValueError(f"Excess Kurtosis, value of: {kurtosis}. Thickness map not plotted.")
        else:
            cv2.imwrite(
                f"{save_dir}/results/foveal_total_thickness_maps/{save_prefix}foveal_thickness_map.png",
                t_u16
            )  # cv2 correctly saves 16-bit 
    
        return t_u16
    
    def qc_overlay_slice(self, slice_idx, fov_loc, centre_y, rot_deg, save_dir="./", save_prefix="testing", mode="unsmoothed"):# this is just to provide a version with segmentation overlays and the foveal location for the purpose of manually interpreting outputs / as a sanity check
        """
        oct: OCTVolumeWithMetaData instance
        slice_idx: which B-scan
        fov_loc, centre_y: rotation center (pixel coords) used in your rotation
        rot_deg: the rotation angle you applied (degrees)
        """
        sx_mm, _, _, _, _, _, _, _, _, _, _, _ = self.derive_scales_and_coord_systems()
        keys = set(self.contours.keys())
        ilm_key = "ILM" if "ILM" in keys else ("RETINA_1" if "RETINA_1" in keys else None)
        bm_key  = "BM"  if "BM"  in keys else ("RETINA_4" if "RETINA_4" in keys else None)
        if ilm_key is None or bm_key is None:
            raise ValueError(f"Missing ILM/BM. Have: {sorted(keys)}")
    
        img = self.volume[slice_idx]
        ilm = self.contours[ilm_key][slice_idx]
        bm  = self.contours[bm_key][slice_idx]
    
        H, W = img.shape[:2]
        ilm = np.asarray(ilm, dtype=np.float32).reshape(-1)
        bm  = np.asarray(bm,  dtype=np.float32).reshape(-1)
        if ilm.shape[0] != W or bm.shape[0] != W:
            raise ValueError(f"Boundary length mismatch: W={W}, ilm={ilm.shape}, bm={bm.shape}")
    
        center = (float(fov_loc), float(centre_y))
    
        rot_img, M, (ilm_xr, ilm_yr), (bm_xr, bm_yr) = rotate_image_and_boundaries(
            img,sx_mm, ilm, bm, center_xy=center, rot_deg=rot_deg, mode=mode
        )
    
        # draw overlays
        over = draw_polyline_points(rot_img, ilm_xr, ilm_yr, color=(0, 255, 0), radius=1)  # ILM green
        over = draw_polyline_points(over,     bm_xr,  bm_yr,  color=(0, 0, 255), radius=1)  # BM red
    
        # mark rotation center
        over = over.copy()
        cv2.drawMarker(over, (int(round(center[0])), int(round(center[1]))),
                      color=(0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=12, thickness=2)
    
        # show/save
        fig,ax = plt.subplots(1, 2, figsize=(6,4)) 
        #plt.figure(figsize=(8, 6))
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title(f"Original Central Slice", fontsize=10)
        ax[0].axis("off")
    
        ax[1].imshow(over[..., ::-1])  # BGR->RGB for matplotlib
        ax[1].set_title(f"Rotated & Segmented Central Slice", fontsize=10)
        ax[1].axis("off")    
        fig.suptitle(f"Slice {slice_idx} | Rotation={rot_deg:.2f}°")
        plt.tight_layout()
        if save_prefix is not None:
            plt.savefig(f"{save_dir}/results/overlays/{save_prefix}{mode}_rotated_segmentation_overlay.png", dpi=400)
            plt.close()
        else:
            plt.show()
            
def cart2pol(x, y):# Input cartesian, output polar coordinates. Works element-wise if arrays are passed (important but unstated).
    # Compute the angle θ of the point relative to the positive x-axis.
    theta = np.arctan2(y, x)# Derive angle from a reference direction (angle), "element-wise arc tangent of x1/x2 choosing the quadrant correctly" (x1 here is y and x2 is y).
    rho = np.hypot(x, y)# Derived the distance from the origin - it returns the hypotenuse of a triangle (ie equivalent to sqrt(x1**2 + x2**2)). In polar coordinates, ρ (rho) is the radial coordinate: the distance from the origin to a point.
    return theta, rho
    
def find_min(X,Y,Z):
    # First, find the min value with argmin. 
    ind_min = np.nanargmin(Z)# Return the indices of the minimum values in the specified axis ignoring NaNs. numpy.nanargmin(a, axis=None, out=None, *, keepdims=<no value>)
    # argmin returns a flat index, so you need to unravel it to (row,col) coords.
    ind_x, ind_y = np.unravel_index(ind_min, Z.shape)# Converts a flat index or array of flat indices into a tuple of coordinate arrays. numpy.unravel_index(indices, shape, order='C')
    # Use these indices to find the row and col vals (x and y labels are slightly misleading, these arent actually cartesian coords)
    x_min = X[ind_x, ind_y]
    y_min = Y[ind_x, ind_y]
    return x_min, y_min

def apply_affine_to_point(M, x, y):
    """Apply a 2x3 OpenCV affine matrix to a point (x,y)."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    xr, yr = M @ pt
    return float(xr), float(yr)
    
def _mask_between_ilm_bm(ilm_y, bm_y, h, w):
    ilm_y = np.asarray(ilm_y, dtype=np.float32).reshape(-1)
    bm_y  = np.asarray(bm_y,  dtype=np.float32).reshape(-1)
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = np.isfinite(ilm_y) & np.isfinite(bm_y)
    if not np.any(valid):
        return mask

    top = np.clip(ilm_y, 0, h - 1).astype(np.int32)  # distance-from-top
    bot = np.clip(bm_y,  0, h - 1).astype(np.int32)

    xs = np.where(valid & (bot >= top))[0]
    for x in xs:
        mask[top[x]:bot[x] + 1, x] = 1
    return mask

#def is_valid_map(x, min_fraction=0.95):
#    finite = np.isfinite(x)
#    return finite.any() and (finite.mean() >= min_fraction)


def rotate_pit_to_bm_vertical(img, ilm_y, bm_y, fov_loc, sx_mm, sy_mm,
                              window_mm=1.0):
    """
    Rotate B-scan so that the shortest vector from the foveal pit base (ILM minimum)
    to the BM is vertical (perpendicular to the bottom edge).

    img: (H,W) or (H,W,C)
    ilm_y: length W ILM y-coordinates in pixels (distance-from-top)
    bm_y:  length W BM  y-coordinates in pixels (distance-from-top)
    fov_loc: approximate fovea x index (column)
    sx_mm: mm per x pixel (A-scan spacing)
    sy_mm: mm per y pixel (axial spacing)
    window_mm: half-window in mm around fovea to search for pit base + closest BM (1mm should be plenty) 
    """
    H, W = img.shape[:2]# typically 650, 512 for a topcon fda (at least for the early instances)
    sigma_px = 0.3 / sx_mm
    if ilm_y.shape[0] != W or bm_y.shape[0] != W:
        raise ValueError("ilm_y and bm_y must have length equal to image width.")# this is to error the script if you have some silly values
    
    # sav gol smoothing to stop BM outliers from affecting closest single point
    L = int(round(0.5 / sx_mm))
    L = max(L, 5)
    if L % 2 == 0: L += 1
    good = np.isfinite(bm_y)

    bm_y_2 = bm_y.copy()
    if good.sum() >= 2:
        x = np.arange(bm_y.shape[0])
        bm_y_2[~good] = np.interp(x[~good], x[good], bm_y_2[good])
        bm_y = gaussian_filter1d(bm_y_2, sigma=sigma_px, mode='nearest')
        #bm_y = savgol_filter(bm_y_2, window_length=L, polyorder=2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        bm_y[~good] = np.nan
    else:
        # not enough points to fill/smooth
        return img, 0.0, fov_loc, np.nan

    # window in x pixels
    half_w_px = int(round(window_mm / sx_mm))# The 'window' defines where candidate BM areas are
    half_w_px = max(5, half_w_px)  # ensure enough picels to select a candidate- really there should be plenty of picels here so this shouldnt be needed
    #print(f"Total eligible pixels is : {half_w_px}")
    a1 = max(0, fov_loc - half_w_px)# the LEFT image side after windowing - not allowing you to do a crop that tries to go to outside the image border
    a2 = min(W, fov_loc + half_w_px + 1)# the RIGHT image side after window - again not allowing to outside the border

    bm_win_y  = bm_y[a1:a2]# crop to required window
    x1 = np.arange(a1, a2, dtype=np.int32)# xs will be evenly spaced piuxel values between the upper and lower x axis range
    
    # valid masks
    good_bm  = np.isfinite(bm_win_y)


    # pick pit base point (x0, y0) from ILM (this is just based on the fovea finder function values)
    x0 = fov_loc
    y0 = float(ilm_y[x0])
    if good_bm.sum() < 3:
        return img, 0.0, fov_loc, y0
    x1 = x1[good_bm]
    y1 = bm_win_y[good_bm].astype(np.float32)  
    
    # distances in physical space rather than image space becase the axial and width pixel spacing is differnet
    x_diffs = (x1 - x0) * sx_mm
    y_diffs = (y1 - y0) * sy_mm
    
    # get an array of total distance (pythagoras theorem minus sqroot because its not needed
    pythag_diff = x_diffs**2 + y_diffs**2
    
    # now get the index of the smallest distance
    min_index = np.argmin(pythag_diff)
    
    # now get the y-coord of the closest point
    min_val_bm_y = float(y1[min_index])
    min_val_bm_x = int(x1[min_index])
    
    # vector from pit -> closest BM point (in mm)
    vx = (min_val_bm_x - x0) * sx_mm
    vy = (min_val_bm_y - y0) * sy_mm

    # If vy ~ 0 and vx ~ 0 (degenerate), bail
    if (vx*vx + vy*vy) < 1e-12:
        return img, 0.0, fov_loc, y0

    # rotate so this vector is vertical.
    
    # Current vector angle (relative to +x axis):
    theta = np.arctan2(vy, vx)  # radians
    
    # Desired vertical direction is +y axis => angle = +pi/2
    rot_rad = (np.pi/2) - theta
    rot_deg = np.degrees(rot_rad)

    # Rotate about the pit base point (x0, y0) so it stays centered. In testing i found it sometimes rotates the wrong direction, so we rotate both ways and empirically test which minimises the angle of the ILM-BM vector at the foveal pit.
    center = (float(x0), float(y0))
    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)# Positive rotation
    M2 = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)# Negative rotation

    # Identify which direction of rotation is required to get the ILM --> BM to 90 degrees
    rotated = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    rotated_2 = cv2.warpAffine(
        img, M2, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    # Transform the BM reference point under each rotation
    xb1, yb1 = apply_affine_to_point(M,  min_val_bm_x, min_val_bm_y)
    xb2, yb2 = apply_affine_to_point(M2, min_val_bm_x, min_val_bm_y)

    # Compare direction using angle-to-vertical ---
    # After rotation, the vector from (x0,y0) to (xb',yb') should be vertical => x component ~ 0.
    vx1 = (xb1 - x0) * sx_mm
    vy1 = (yb1 - y0) * sy_mm
    ang1 = abs(np.degrees(np.arctan2(vx1, vy1)))  # 0 means perfectly vertical

    vx2 = (xb2 - x0) * sx_mm
    vy2 = (yb2 - y0) * sy_mm
    ang2 = abs(np.degrees(np.arctan2(vx2, vy2)))  # 0 means perfectly vertical

    # Choose the rotation with smaller angle-to-vertical
    if ang2 < ang1:
        return rotated_2, -rot_deg, x0, y0
    else:
        return rotated, rot_deg, x0, y0

def rotate_around_point_highperf(x, y, radians, origin=(0, 0)):# taken from https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    Rotate point(s) (x,y) about origin by 'radians'.

    """
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + (-sin_rad) * adjusted_x + cos_rad * adjusted_y
    return qx, qy


def _rotate_curve_and_resample_to_xgrid(y, angle_rad, origin_xy):# this code serves the objective of rotating the thickness map around the foveal pit ILM-BM normal - this aims to stop extreme rotation causing error
    """
    y: array (w,) where y[x] is the boundary (ILM/BM) in pixel coords.
    Rotate the polyline points (x, y[x]) about origin_xy,
    then resample back onto integer x-grid 0..w-1.

    Returns y_rot_on_grid (w,) with NaNs where undefined.
    """
    y = np.asarray(y, dtype=np.float32)# this is redundant really as this is enforced in the thickness map code
    if y.ndim != 1:
        raise ValueError(f"`y` must be 1D (shape (w,)), got shape {y.shape}")
    w = y.shape[0]# to return the width of the y var (the length of the vector)
    x = np.arange(w, dtype=np.float32)# becuase the x coords are just 0-width (as thats how y has been orgnaised in the fda file)

    good = np.isfinite(y)# this returns a boolean array for finite nature of vals (not an NaN and not infinite) 
    if good.sum() < (w*0.9):# true is 1 and false is 0, so this is saying if we have < 90% real vals nan the whole thing 
        return np.full(w, np.nan, dtype=np.float32)

    qx, qy = rotate_around_point_highperf(x[good], y[good], angle_rad, origin=origin_xy)# run the valid x and y thru the rotation function to rotate new coords

    qx = np.asarray(qx, dtype=np.float32)# re assert array nature (probably not needed)
    
    qy = np.asarray(qy, dtype=np.float32)# re assert array nature (probably not needed)

    # sort by rotated x for interpolation
    order = np.argsort(qx)# returns the *indices* that would sort an array - its possible that some x near the origin flip order
    qx = qx[order]# re-sort according to a new order (probably wont actually change for images except the absolute extreme of rotation)
    qy = qy[order]

    xg = np.arange(w, dtype=np.float32)# creates a 0-width array 
    out = np.full(w, np.nan, dtype=np.float32)# creates an array of nans of width image size - this is a prepared output grid filled with NaNs

    lo, hi = float(qx[0]), float(qx[-1])# this tells you the minimum new x coordinate and the maximum new x coordinate (as they have been ordered 0-high above) 
    inside = (xg >= lo) & (xg <= hi)# returns a boolean indicating which vals are captured within the new val map (so you know what needs filling with NaNs -  as we dont want to interpolate outside of the range of values - anything outside lo and hi just needs to be an NaN (i.e. dont extrapolate))
    if inside.any():
        out[inside] = np.interp(xg[inside], qx, qy).astype(np.float32) # np.interp - one-dimensional linear interpolation for monotonically increasing sample point - this helps us to convert non integer non 0-w x vales to integer 0-w and interpolate y as needed
        #numpy.interp(x, xp, fp, left=None, right=None, period=None) - x is the x coords at which the evaluate interpolated values, xp is the x coords of the data points and must be increasing, fp is the y coordinates of the data points 
    return out

def apply_affine_to_points(M, xs, ys):
    """
    M: 2x3 affine matrix from cv2.getRotationMatrix2D - ie the means of rotating a set of pixel coord in opencv
    xs, ys: 1D arrays of equal length, in pixel coords.
    Returns: (x', y') arrays (float32)
    """
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)  # (N,3)
    out = (M @ pts.T).T  # (N,2)
    return out[:, 0].astype(np.float32), out[:, 1].astype(np.float32)

def draw_polyline_points(img_u8, xs, ys, color=(0, 255, 0), radius=1, valid_mask=None):
    """
    Draw points (xs, ys) onto img_u8 (grayscale or BGR).
    If img is grayscale, it will be converted to BGR for colored overlay.
    """
    if img_u8.ndim == 2:
        canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_u8.copy()

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if valid_mask is None:
        valid_mask = np.isfinite(xs) & np.isfinite(ys)

    h, w = canvas.shape[:2]
    # also require inside image bounds
    valid_mask = valid_mask & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    xi = xs[valid_mask].astype(np.int32)
    yi = ys[valid_mask].astype(np.int32)

    for x, y in zip(xi, yi):
        cv2.circle(canvas, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas

def rotate_image_and_boundaries(img, sx_mm, ilm_y, bm_y, center_xy, rot_deg, border_value=0, mode="unsmoothed"):
    """
    Rotate image and transform ILM/BM points using the same affine matrix.
    img: (H,W) uint8 or float; we'll convert to uint8 for display.
    ilm_y, bm_y: length W arrays (y per x).
    center_xy: (cx, cy) rotation center in pixel coords.
    rot_deg: degrees
    """
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D(center_xy, rot_deg, 1.0)# this gives you an affine matrix for rotation of pix coords
    sigma_px = 0.3/sx_mm
    # rotate image
    if img.dtype != np.uint8:
        # for QC only: scale to 0..255 robustly
        im = img.astype(np.float32)
        lo, hi = np.nanpercentile(im, [1, 99])
        im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
        img_u8 = (im * 255).astype(np.uint8)
    else:
        img_u8 = img

    rot = cv2.warpAffine(
        img_u8, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    # transform boundary points as points (x, y[x])
    x = np.arange(W, dtype=np.float32)
    ilm_y = np.asarray(ilm_y, dtype=np.float32)
    bm_y  = np.asarray(bm_y,  dtype=np.float32)

    good_ilm = np.isfinite(ilm_y)
    good_bm  = np.isfinite(bm_y)
    
    if mode == "smoothed":
        L = int(round(0.5 / sx_mm))
        L = max(L, 5)
        #if L % 2 == 0: L += 1
        bm_y_2 = bm_y.copy()
        if good_bm.sum() >= 2:
            bm_y_2[~good_bm] = np.interp(x[~good_bm], x[good_bm], bm_y_2[good_bm])
            bm_y = gaussian_filter1d(bm_y_2, sigma=sigma_px, mode='nearest')
            #bm_y = savgol_filter(bm_y_2, window_length=L, polyorder=2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
            bm_y[~good_bm] = np.nan
            ilm_xr, ilm_yr = apply_affine_to_points(M, x[good_ilm], ilm_y[good_ilm])
            bm_xr,  bm_yr  = apply_affine_to_points(M, x[good_bm],  bm_y[good_bm])
        else:
            # not enough points to fill/smooth
            raise ValueError(f"You dont have enough points for smoothing.")

    elif mode == "unsmoothed":
        ilm_xr, ilm_yr = apply_affine_to_points(M, x[good_ilm], ilm_y[good_ilm])
        bm_xr,  bm_yr  = apply_affine_to_points(M, x[good_bm],  bm_y[good_bm])    

    return rot, M, (ilm_xr, ilm_yr), (bm_xr, bm_yr)

def compute_features(detail_coeffs):
    features = []
    for cH, cV, cD in detail_coeffs:
        #print(cH.shape)
       # Convert absolute coefficients to "probabilities" by normalizing
        cH_abs = np.abs(cH)
        cV_abs = np.abs(cV)
        cD_abs = np.abs(cD)

        # Avoid division by zero
        pH = cH_abs / (np.sum(cH_abs) + 1e-10)
        pV = cV_abs / (np.sum(cV_abs) + 1e-10)
        pD = cD_abs / (np.sum(cD_abs) + 1e-10)
       
        # Energy: Sum of squared coefficients
        energy = np.sum(cH**2)# + np.sum(cV**2) + np.sum(cD**2)
        energy_2 = np.sum(cV**2)
        # Shannon entropy (non-negative)
        entropy = -np.sum(pH * np.log(pH + 1e-10)) #\
                  #-np.sum(pV * np.log(pV + 1e-10)) \
                  #-np.sum(pD * np.log(pD + 1e-10))

        # Mean of absolute coefficients
        H_V_ratio = energy/(energy_2 + 1e-10)
        kurt_H = scipy.stats.kurtosis(cH.ravel(), fisher=True)
        features.append((energy, entropy, H_V_ratio,kurt_H))
        
    return features
    
def energy_quantification(image, save_name):
    # Perform multi-level wavelet decomposition
    #image = Image.open(image)
    #image.load()
    arr = np.array(image).astype(np.float32) / 65535

    wavelet = 'db4'  # Daubechies wavelet
    level = 5  # Level of decomposition , each level works at 2x less resolution
    coeffs = pywt.wavedec2(arr, wavelet, level=level)

    # Separate approximation and detail coefficients
    cA, *detail_coeffs = coeffs

    # Compute features for detail coefficients
    features = compute_features(detail_coeffs)
    #print(np.size(features))

    # Display features for each level (highest level first)
    #for i, (energy, entropy, mean) in enumerate(features):
    #    print(f"Level {level - i}:")  # Reverse the numbering
    #    print(f"  Energy: {energy:.2f}")
    #    print(f"  Entropy: {entropy:.2f}")
    #    print(f"  Mean: {mean:.2f}")

    # Visualize energy maps for each level (highest level first)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(detail_coeffs)+1,1)
    plt.imshow(arr,cmap='gray')
    plt.title(f"Original Image")
    plt.axis("off")
    for i, (cH, cV, cD) in enumerate(detail_coeffs):
        # Compute energy map
        energy_map = cH**2# + cV**2 + cD**2
        plt.subplot(1, len(detail_coeffs)+1, i + 2)
        plt.imshow(energy_map, cmap='hot', aspect="equal")
        plt.title(f"Energy Map - Level {level - i}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_name, dpi=400)
    plt.close()
    return features# ordered by - energy [0] - entropy [1] - HV Ratio [2] and Kurtosis [3]; levels are organised 5-1 (so, level 4 which you are interested in, is [1] ; therefore features[1][3] is what we want to filter on for kurtosis level 4 
