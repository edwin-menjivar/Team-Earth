"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/10/2022

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Tags:

Reference:

"""
import json
import os

import SimpleITK as sitk

import util
import wrapper_segmentation

PATH_EXAMPLE_VOLUME = r"H:\Datasets\zenodo\covid_19_ct_lung_and_infection_segmentation_dataset\data\coronacases_001.nii.gz"

PATH_INPUT = os.path.join(os.getcwd(), "input")
PATH_OUTPUT = os.path.join(os.getcwd(), "output")


def write(file_path_abs_output: str, sitk_image: sitk.Image) -> None:
    wrapper_segmentation.write_sitk_image_to_file_path(
        sitk_image,
        file_path_abs_output
    )


def operation(file_path_abs: str, file_filename_path_relative: str) -> None:
    (
        sitk_image_covid_19_segmentation_ct_lung_seg,
        sitk_image_lung_extracted_ct_lung_seg,
        sitk_image_covid_19_extracted,
        np_ndarray_lung_segmentation_lungmask,
        model_handler,
        sitk_image_original
    ) = wrapper_segmentation.get_sitk_image_covid_19_segmentation_ct_lung_seg_using_R231CovidWeb_all_in_one(
        file_path_abs
    )
    #####

    # path_output = os.path.join(PATH_OUTPUT, file_filename_path_relative)
    path_output = os.path.join(PATH_OUTPUT)

    #####

    filename_sitk_image_lung_extracted_ct_lung_seg = file_filename_path_relative + "_lung_extracted.nii"
    path_abs_sitk_image_lung_extracted_ct_lung_seg = os.path.join(path_output,
                                                                  filename_sitk_image_lung_extracted_ct_lung_seg)

    write(path_abs_sitk_image_lung_extracted_ct_lung_seg, sitk_image_lung_extracted_ct_lung_seg)

    #####

    filename_sitk_image_covid_19_extracted = file_filename_path_relative + "_covid_19_extracted.nii"
    path_abs_sitk_image_covid_19_extracted = os.path.join(path_output,
                                                          filename_sitk_image_covid_19_extracted)
    write(path_abs_sitk_image_covid_19_extracted, sitk_image_covid_19_extracted)

    #####

    np_ndarray_covid_19_segmentation_ct_lung_seg = sitk.GetArrayFromImage(
        sitk_image_covid_19_segmentation_ct_lung_seg
    )

    severity_score, area_lung_total, area_covid_19_total = util.calculate_count_region_sub_over_count_region_total(
        np_ndarray_lung_segmentation_lungmask,
        np_ndarray_covid_19_segmentation_ct_lung_seg
    )

    dict_severity_score = {
        "severity_score": severity_score,
        "area_lung_total": int(area_lung_total),
        "area_covid_19_total": int(area_covid_19_total)
    }

    filename_severity_score = file_filename_path_relative + "_severity_score.json"
    path_abs_filename_severity_score = os.path.join(path_output,
                                                    filename_severity_score)

    with open(path_abs_filename_severity_score, 'w') as outfile:
        json.dump(dict_severity_score, outfile)


def run() -> None:
    list_file_input = os.listdir(PATH_INPUT)

    list_dir_output = os.listdir(PATH_OUTPUT)

    set_dir_output_dir = set(dir_ for dir_ in list_dir_output if os.path.isdir(dir_))

    for file_input in list_file_input:

        file_input_filename_path_relative = os.path.splitext(file_input)[0]

        file_input_path_abs = os.path.join(PATH_INPUT, file_input)

        if (file_input_filename_path_relative not in set_dir_output_dir):

            operation(file_input_path_abs, file_input_filename_path_relative)

        else:
            print(f"Dir for {file_input_filename_path_relative} already exists")

    print("Done.")


if __name__ == '__main__':
    run()
