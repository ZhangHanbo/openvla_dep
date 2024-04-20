from pathlib import Path
import os
import random
import numpy as np
import tqdm
import base64
import requests
from PIL import Image
import warnings
import numpy as np
import requests
from image_utils import *
from lang_sam import LangSAM
import torch
import logging

def segement_with_prompt(
    model: LangSAM, image_pil: Image, 
    text_prompt: str="Object caught in the jaws",
    background_prompts: str = "A pair of white clamping jaws on the machine arm",
    box_threshold: float = 0.3, text_threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    segement the object by the text prompt and remove the background to improve the segementation by the background prompt
    args:
        model: LangSAM, the model to predict the object in the image
        image_pil: Image, the image to predict the object
        text_prompt: str, the text prompt to detect the object in the image
        background_prompts: str, the text prompt to detect the background in the image
        box_threshold: float, the threshold to filter the bbox
        text_threshold: float, the threshold to filter the text
    return:
        masks: np.ndarray, [bbox_number, h, w]
        logits: np.ndarray, [bbox_number, 256, 256], the output of the sam which predict the pixel-level mask of the object
        boxes: np.ndarray, [bbox_number, 4], bbox[0] is the min width, bbox[1] is the min height, bbox[2] is the max width, bbox[3] is the max height
        phrases: list[str], len = bbox_number, it represents the detected object classes of each bbox
        scores: np.ndarray, [bbox_number], the confidence score of each bbox
        background_mask_np: np.array, [h, w], the mask of the background that best matched the background prompt
        background_points: np.array, [k, 2], note that the coord is (w, h), not the same with the image array with (h, w). 
            It represents the points in the background mask to mark as the background when predicting the sam by text prompt
    """
    background_masks, _, _, background_scores = model.predict(image_pil, background_prompts)
    if len(background_masks) == 0:
        print(f"No objects of the '{background_prompts}' prompt detected in the image.")
        return np.empty((0, image_pil.height, image_pil.width)), np.empty((0, 256, 256)), np.empty((0, 4)), [], np.empty(0), np.empty((image_pil.height, image_pil.width)), np.empty((0, 2))
    background_mask_np = background_masks[background_scores.argmax()].squeeze().cpu().numpy()
    background_points = sample_points_from_mask(background_mask_np)
    boxes, scores, phrases = model.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
    masks = torch.empty(0, image_pil.height, image_pil.width)    
    logits = torch.empty(0, 256, 256)
    if len(boxes) > 0:
        image_array = np.asarray(image_pil)
        transformed_boxes = model.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        transformed_points = model.sam.transform.apply_coords_torch(torch.tensor(background_points), image_array.shape[:2])
        transformed_points = transformed_points[None].repeat(len(boxes), 1, 1)
        point_labels = torch.tensor([0] * len(background_points))
        point_labels = point_labels[None].repeat(len(boxes), 1)
        masks, _, logits = model.sam.predict_torch(
            point_coords=transformed_points.to(model.sam.device),
            point_labels=point_labels.to(model.sam.device),
            boxes=transformed_boxes.to(model.sam.device),
            multimask_output=False,
        )
        masks = masks.cpu().squeeze(1).numpy()
    return masks, logits.cpu().numpy(), boxes.cpu().numpy(), phrases, scores.cpu().numpy(), background_mask_np, np.array(background_points)


def segement_by_sam(
    model: LangSAM, 
    image_pil: Image,
    logits: np.ndarray=None,
    bbox: tuple[int, int, int, int]=None,
    background_prompts: str = "A pair of white clamping jaws on the machine arm",
) -> tuple[np.ndarray, np.ndarray]:
    """
    segement the object by the mask of neighbor image and remove the background to improve the segementation by the background prompt
    args:
        model: LangSAM, the model to predict the object in the image
        image_pil: Image, the image to predict the object
        logits: np.ndarray, [h, w], the mask of the object in the neighbor image
        bbox: tuple[int, int, int, int], the bbox of the object in the neighbor image
        background_prompts: str, the text prompt to detect the background in the image
    return:
        mask: np.ndarray, [h, w], the mask of the object in the image
        logit: np.ndarray, [256, 256], the pixel-level logit of the above mask, [
    """
    background_masks, _, _, background_scores = model.predict(image_pil, background_prompts)
    model.sam.set_image(np.asarray(image_pil))
    if len(background_masks) == 0:
        print(f"No objects of the '{background_prompts}' prompt detected in the image.")
        return np.empty((image_pil.height, image_pil.width)), np.empty((256, 256))
    background_mask_np = background_masks[background_scores.argmax()].squeeze().cpu().numpy()
    background_points = sample_points_from_mask(background_mask_np)
    # TODO delete the trash code
    # masks, _, logits = model.sam.predict_torch(
    #     point_coords=torch.tensor(background_points).to(model.sam.device)[None],
    #     point_labels=torch.tensor([0] * len(background_points)).to(model.sam.device)[None].repeat(1, 1),
    #     mask_input=torch.from_numpy(logits).unsqueeze(0).to(model.sam.device),
    #     multimask_output=False
    # )
    masks, _, logits = model.sam.predict(
        point_coords=np.array(background_points),
        point_labels=np.array([0] * len(background_points)),
        box=np.array(bbox) if bbox is not None else None,
        mask_input=logits,
        multimask_output=False
    )
    return masks[0], logits[0]
