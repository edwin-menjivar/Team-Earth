"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/9/2022

Purpose:
    Fully automatic lung segmentation

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Tags:

Reference:
    JoHof/lungmask
        Reference:
            https://github.com/JoHof/lungmask/tree/master/lungmask

    RiccardoBiondi/segmentation
        Reference:
            https://github.com/RiccardoBiondi/segmentation

"""
from __future__ import annotations

from typing import Dict, List, Tuple

import SimpleITK as sitk
import numpy as np

import handler_lungmask

"""
Lung mask library

Notes:
    
Reference:
    JoHof/lungmask
        Notes:
            Installation
                pip install git+https://github.com/JoHof/lungmask

        Reference:
            https://github.com/JoHof/lungmask/tree/master/lungmask
    
"""
from lungmask import mask as lungmask_mask

# from lungmask import mask as lungmask_resunet
# from lungmask import mask as lungmask_utils

"""

Reference:
    RiccardoBiondi/segmentation
        Notes:
            Installation
                git clone https://github.com/RiccardoBiondi/segmentation
                pip install segmentation/
            
            The package name is called CTLungSeg
    
        Reference:
            https://github.com/RiccardoBiondi/segmentation
"""
from CTLungSeg import utils as ct_lung_seg_utils
from CTLungSeg import labeling as ct_lung_seg_labeling
# from CTLungSeg import lung_extraction as ct_lung_lung_extraction
from CTLungSeg import method as ct_lung_seg_method
from CTLungSeg import segmentation as ct_lung_seg_segmentation

# from CTLungSeg import __main__ as ct_lung_seg_main


PATH_EXAMPLE = r"H:\Datasets\figshare\covid_rate\COVID-Rate-CT\p007\p07im004.dcm"


def get_sitk_image_from_path(path: str) -> sitk.Image:
    """

    Notes:
        WIth the reader, you get all the image/images

    :param path:
    :return:
    """

    return ct_lung_seg_utils.read_image(path)


def get_np_ndarray_segmentation_lungmask(model_handler: handler_lungmask.ModelHandler,
                                         sitk_image: sitk.Image) -> np.ndarray:
    """
    Get np ndarray segmentation

    :param model_handler:
    :param sitk_image:
    :return:
    """

    np_array_segmentation_lungmask: np.ndarray = lungmask_mask.apply(sitk_image, model_handler.model)

    return np_array_segmentation_lungmask


def get_sitk_image_copy_sitk_image_information(np_ndarray_given: np.ndarray,
                                               sitk_image_given: sitk.Image) -> sitk.Image:
    # Copy image information from sitk_image_lung_extracted to sitk_image_from_np_array_new
    sitk_image_from_np_array_new: sitk.Image = sitk.GetImageFromArray(np_ndarray_given)
    sitk_image_from_np_array_new.CopyInformation(sitk_image_given)

    return sitk_image_from_np_array_new


def get_sitk_image_lung_extracted_ct_lung_seg(np_ndarray_given: np.ndarray,
                                              sitk_image_given: sitk.Image) -> sitk.Image:
    """
    Partial copied code segment from https://github.com/RiccardoBiondi/segmentation/blob/0e75911ff56fbc87bb2cdfaa8212a7f8206aa1ae/CTLungSeg/lung_extraction.py#L1


    Notes:
        Process:
            1.  Make np_ndarray_given into another np ndarray that is binary with the segmentation as an unsigned 8 bit int
            2.  Copy information about the images used to make np_ndarray_given from what created it which was
                sitk_image_lung_extracted
            3.

        The code below mimics the function
            lung = lung_extraction.main(volume)

        from https://github.com/RiccardoBiondi/segmentation/blob/master/CTLungSeg/__main__.py

    :param np_ndarray_given:
    :param sitk_image_given:
    :return:
    """

    """
    
    Notes:
        Process:
            1. Get np_ndarray_given
            2. Make it to a boolean ndarray
            3. Cast the new ndarray as Numpy unsigned 8 bit int  
    """
    np_array_new: np.ndarray = (np_ndarray_given != 0).astype(np.uint8)

    sitk_image_from_np_array_new: sitk.Image = get_sitk_image_copy_sitk_image_information(
        np_array_new,
        sitk_image_given
    )

    sitk_image_masked: sitk.Image = ct_lung_seg_method.apply_mask(
        image=sitk_image_given,
        mask=sitk_image_from_np_array_new,
        outside_value=-1000
    )

    sitk_image_masked_no_vessels: sitk.Image = ct_lung_seg_segmentation.remove_vessels(
        image=sitk_image_masked)

    sitk_image_masked_no_vessels_shift_crop: sitk.Image = ct_lung_seg_utils.shift_and_crop(
        image=sitk_image_masked_no_vessels)

    return sitk_image_masked_no_vessels_shift_crop


def get_sitk_image_segmentation_mask_covid_ct_lung_seg_from_sitk_image_lung_extracted(
        sitk_image_lung_extracted: sitk.Image,
        center: Dict[List] = None) -> sitk.Image:
    """
        Partial copied code segment from https://github.com/RiccardoBiondi/segmentation/blob/master/CTLungSeg/__main__.py

    :param sitk_image_lung_extracted:
    :param center:
    :return:
    """

    if center is None:
        np_ndarray_center = np.asarray([np.array(v) for _, v in ct_lung_seg_labeling.centroids.items()])
    else:
        np_ndarray_center = center

    sitk_image_segmentation_covid = ct_lung_seg_labeling.main(
        sitk_image_lung_extracted,
        np_ndarray_center
    )

    return sitk_image_segmentation_covid


def write_sitk_image_to_file_path(sitk_image_given: sitk.Image, filename: str) -> None:
    ct_lung_seg_utils.write_volume(sitk_image_given, filename)


####################################################################################################


def _get_model_handler_R231CovidWeb():
    dict_k_model_name_v_model_handlers = handler_lungmask.get_dict_k_model_name_v_model_handlers()

    model_handler_R231CovidWeb = dict_k_model_name_v_model_handlers.get('R231CovidWeb')

    return model_handler_R231CovidWeb


def get_sitk_image_segmentation_mask_lung_lungmask_using_R231CovidWeb(path_image: str) -> sitk.Image:
    sitk_image_original = get_sitk_image_from_path(path_image)

    model_handler = _get_model_handler_R231CovidWeb()

    #####

    np_array_segmentation_lungmask = get_np_ndarray_segmentation_lungmask(model_handler, sitk_image_original)

    sitk_image_segmentation_lung = get_sitk_image_copy_sitk_image_information(
        np_array_segmentation_lungmask,
        sitk_image_original
    )

    return sitk_image_segmentation_lung


def get_sitk_image_segmentation_mask_covid_ct_lung_seg_using_R231CovidWeb_all_in_one(
        path_image: str) -> Tuple[sitk.Image, sitk.Image, np.ndarray, handler_lungmask.ModelHandler, sitk.Image]:
    sitk_image_original = get_sitk_image_from_path(path_image)

    model_handler = _get_model_handler_R231CovidWeb()

    #####

    np_ndarray_segmentation_lungmask = get_np_ndarray_segmentation_lungmask(model_handler, sitk_image_original)

    sitk_image_lung_extracted_ct_lung_seg = get_sitk_image_lung_extracted_ct_lung_seg(
        np_ndarray_segmentation_lungmask,
        sitk_image_original
    )

    sitk_image_segmentation_lung_ct_lung_seg = get_sitk_image_segmentation_mask_covid_ct_lung_seg_from_sitk_image_lung_extracted(
        sitk_image_lung_extracted_ct_lung_seg,
        None
    )

    return (sitk_image_segmentation_lung_ct_lung_seg,
            sitk_image_lung_extracted_ct_lung_seg,
            np_ndarray_segmentation_lungmask,
            model_handler,
            sitk_image_original)


def get_sitk_image_segmentation_mask_covid_ct_lung_seg_using_R231CovidWeb(path_image: str) -> sitk.Image:
    return get_sitk_image_segmentation_mask_covid_ct_lung_seg_using_R231CovidWeb_all_in_one(path_image)[0]


if __name__ == '__main__':
    pass
