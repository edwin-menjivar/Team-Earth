"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/9/2022

Purpose:
    Simple lungmask model handler

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Tags:

Reference:

"""
from typing import Dict, Tuple, Union

from lungmask import mask as lungmask_mask
from lungmask import resunet as lungmask_resunet


class ModelHandler:

    def __init__(self, name: str, type_: str, model: Union[lungmask_resunet.UNet]):
        self.name = name
        self.type = type_
        self.model: Union[lungmask_resunet.UNet] = model

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{ModelHandler.__name__} for Model {self.name}"

    def get_model_str(self) -> str:
        return str(self.model)

    def get_model(self) -> Union[lungmask_resunet.UNet]:
        return self.model

    def get_name(self) -> str:
        return self.name


def get_lungmask_dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes() -> Dict[Tuple[str, str],
                                                                                          Tuple[str, int]]:
    dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes = lungmask_mask.model_urls

    return dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes


def get_lungmask_dict_k_tuple_model_type_model_name_v_model() -> Dict[Tuple[str, str],
                                                                      lungmask_resunet.UNet]:
    dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes = (
        get_lungmask_dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes()
    )

    dict_k_tuple_model_type_model_name_v_model = {
        key: lungmask_mask.get_model(*key) for key, value in
        dict_k_tuple_model_type_model_name_v_tuple_model_url_n_classes.items()
    }

    return dict_k_tuple_model_type_model_name_v_model


def get_dict_k_model_name_v_model_handlers() -> Dict[str, ModelHandler]:
    dict_k_tuple_model_type_model_name_v_model = get_lungmask_dict_k_tuple_model_type_model_name_v_model()

    return {key[1]: ModelHandler(key[1], key[0], value) for key, value in
            dict_k_tuple_model_type_model_name_v_model.items()}
