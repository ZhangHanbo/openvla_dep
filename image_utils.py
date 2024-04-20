import os
from PIL import Image
from io import BytesIO
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, mask_np in enumerate(masks):
        axes[i+1].imshow(mask_np, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def save_image_with_all_elements(image: Image=None, masks: np.ndarray=None, boxes: np.ndarray = None, background_points: np.ndarray = None, save_path: str|Path = "test.png"):
    fig, ax = plt.subplots()
    ax.axis('off')
    if image is not None:
        ax.imshow(image)
    if masks is not None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = masks.shape[-2:]
        for i in range(len(masks)):
            mask_image = masks[i].reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    if background_points is not None and len(background_points) > 0:
        ax.scatter(background_points[:, 0], background_points[:, 1], color='red', marker='*', s=375, edgecolor='white', linewidth=1.25)   
    plt.savefig(save_path)
    plt.close()

def display_image_with_points(image, coords, marker_size=375):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with BackGround Points")
    ax.axis('off')
    neg_points = coords
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    plt.show()

def print_bounding_boxes(boxes):
    print("Bounding Boxes:")
    for i, box in enumerate(boxes):
        print(f"Box {i+1}: {box}")

def print_detected_phrases(phrases):
    print("\nDetected Phrases:")
    for i, phrase in enumerate(phrases):
        print(f"Phrase {i+1}: {phrase}")

def print_logits(logits):
    print("\nConfidence:")
    for i, logit in enumerate(logits):
        print(f"Logit {i+1}: {logit}")

def sample_points_from_mask(mask: np.ndarray, k: int=2) -> list[tuple[int, int]]:
    # set a vertical middle line in the mask
    middle = mask.shape[1] // 2
    mask[:, middle] = 0
    from skimage import measure
    labels = measure.label(mask, connectivity=1)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) == 0:
        return []
    if len(regions) == 1 or regions[0].area / regions[1].area > 2:
        coords = regions[0].coords
        points = sample_points_from_connectivity_region(coords, mask.shape[0], mask.shape[1], k=k)
    else:
        sample_num1 = (k+1) // 2
        sample_num2 = k - sample_num1
        points1 = sample_points_from_connectivity_region(regions[0].coords, mask.shape[0], mask.shape[1], k=sample_num1)
        points2 = sample_points_from_connectivity_region(regions[1].coords, mask.shape[0], mask.shape[1], k=sample_num2)
        points = np.concatenate([points1, points2], axis=0)
    points = np.concatenate([points[:, 1:2], points[:, 0:1]], axis=1)
    return points.tolist()

def sample_points_from_connectivity_region(coords: np.ndarray, h: int, w: int, kernel_size: int=8, k: int=1) -> np.ndarray:
    """sample k points from the connectivity region

    Args:
        coords (np.ndarray): [points_num, 2]
        h (int): the height of the mask
        w (int): the width of the mask
        kernel_size (int, optional): the size of the kernel to get the average pool. Defaults to 20.
        k (int, optional): the number of points to sample. Defaults to 1.
    """
    mask = np.zeros((h, w))
    # set coords to 1
    mask[coords[:, 0], coords[:, 1]] = 1
    # get the average pooling of the mask
    # stride = 1, padding = True
    mask = np.pad(mask, kernel_size//2, mode='constant', constant_values=0)
    mask = np.array([np.mean(mask[i:i+kernel_size, j:j+kernel_size]) for i in range(h) for j in range(w)])
    mask = mask.reshape(h, w)
    # return the k points with the top k value
    indices = np.argpartition(mask.flatten(), -k)[-k:]
    # 获取这些元素的坐标
    y, x = np.unravel_index(indices, mask.shape)
    return np.stack([y, x], axis=1)


class SimilarityMatrix:
    def __init__(self, image_list: list[np.ndarray], score_list: list[np.ndarray]) -> None:
        """ use the image list and score list to calculate the similarity matrix

        Args:
            image_list (list[np.ndarray]): each element ([image_number, h, w, 3])in the list is a frame of a video, but the frame contain multiple images, only one is the target image
            score_list (list[np.ndarray]): the score list of the image list
        """
        self.frame_index2image_index = {}
        self.image_index2frame_index = {}
        self.image_number_list = []
        count = 0
        for i, image in enumerate(image_list):
            self.image_number_list.append(image.shape[0])
            for j in range(image.shape[0]):
                self.frame_index2image_index[(i, j)] = count
                self.image_index2frame_index[count] = (i, j)
                count += 1
        self.images_array = np.concatenate(image_list, axis=0)
        self.similarity_matrix = self.calculate_similarity(self.images_array)
        self.score_array = np.concatenate(score_list, axis=0)
        self.save_cache_image(image_list)
        
    def save_cache_image(self, images_list: list[np.ndarray]):
        cache_dir = Path('images')
        os.makedirs(cache_dir, exist_ok=True)
        for i, images in enumerate(images_list):
            for j, image in enumerate(images):
                image = Image.fromarray(image)
                image.save(cache_dir / f"{i}_{j}.png")
                
    def visualize_choice(self, images_array: np.ndarray, choice_list: list[int]):
        cache_dir = Path("images") 
        for i, choice in enumerate(choice_list):
            frame_index = (i, choice)
            image_index = self.frame_index2image_index[frame_index]
            image = images_array[image_index]
            image = Image.fromarray(image)
            image.save(cache_dir / f"{i}_choice.png")
    
    
    def calculate_similarity(self, images_array: np.ndarray):
        """get the similarity matrix of the image array

        Args:
            image_array (np.ndarray): [image_number, height, width, 3]

        Returns:
            similarity_matrix (np.ndarray): [image_number, image_number]
        """
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        batch_size = 32
        image_features = []
        for i in range(0, images_array.shape[0], batch_size):
            image_tensor = [
                preprocess(Image.fromarray(image)) 
                for image in images_array[i:i+batch_size]
            ]
            image_tensor = torch.stack(image_tensor).to(device)
            with torch.no_grad():
                batch_image_features = model.encode_image(image_tensor)
                batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
                image_features.append(batch_image_features.cpu().numpy())
        image_features = np.concatenate(image_features, axis=0)
        similarity_matrix = np.empty((image_features.shape[0], image_features.shape[0]))
        for i in range(image_features.shape[0]):
            for j in range(image_features.shape[0]):
                similarity_matrix[i, j] = image_features[i] @ image_features[j]
                similarity_matrix[j, i] = similarity_matrix[i, j]
        return similarity_matrix
    
    def calculate_score(self, image_index: int, alpha: float=0.99) -> float:
        """get the image index to calculate the image similarity score with the other images

        Args:
            image_index (int): the image index in the similarity matrix
            alpha (float, optional): the weight decay of the frame. Defaults to 0.99

        Returns:
            float: the similarity score of the image
        """
        frame_first_index = self.image_index2frame_index[image_index][0]
        frame_number = len(self.image_number_list)
        frame_weight_decay = np.linspace(0, frame_number-1, frame_number)
        frame_weight_decay = np.abs(frame_weight_decay - frame_first_index)
        frame_weight_decay = alpha ** frame_weight_decay
        image2frame_index = [self.image_index2frame_index[i][0] for i in range(len(self.image_index2frame_index))]
        image_weight_decay = frame_weight_decay[image2frame_index]
        score: np.ndarray = image_weight_decay * self.similarity_matrix[image_index] * self.score_array
        return score.sum()
        
    def get_best_choices(self) -> list[int]:
        """ get the best image choice of each frame

        Returns:
            list[int]: the best image choice of each frame
        """
        choice_list = []
        for frame_index in range(len(self.image_number_list)):
            image_number = self.image_number_list[frame_index]
            if image_number == 1:
                choice_list.append(0)
                continue
            # we need to chose the image with the best score
            score_list = [
                self.calculate_score(self.frame_index2image_index[(frame_index, i)])
                for i in range(image_number)
            ]
            choice_list.append(np.argmax(score_list))
        return choice_list
    

class Net:
    def __init__(self, masks_list: list[np.ndarray], scores_list: list[np.ndarray]) -> None:
        """
        args:
            masks: list of masks, each element has the shape [mask_number, height, width]
        """
        self.scores_list = scores_list
        self.layers: list[np.ndarray] = []
        for i in range(len(masks_list)-1):
            masks1 = masks_list[i]
            masks2 = masks_list[i+1]
            similarity_matrix = np.empty((len(masks1), len(masks2)))
            for j in range(len(masks1)):
                for k in range(len(masks2)):
                    similarity_matrix[j, k] = self.calculate_mask_distance(masks1[j], masks2[k])
            self.layers.append(similarity_matrix)
    
    def calculate_mask_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        args:
            mask1: np.ndarray, shape [height, width]
            mask2: np.ndarray, shape [height, width]
        return:
            distance: float
        """
        from scipy.spatial.distance import directed_hausdorff
        distance = directed_hausdorff(mask1, mask2)[0]
        return distance
    
    def get_shorest_path(self):
        # we use dp to solve the problem
        paths = []
        for i in range(len(self.layers)):
            similarity_matrix = self.layers[i]
            if i == 0:
                paths.append(similarity_matrix.argmin(axis=0))
                min_distance = similarity_matrix.min(axis=0)
            else:
                similarity_matrix: np.ndarray = similarity_matrix + min_distance.reshape(-1, 1)
                paths.append(similarity_matrix.argmin(axis=0))
                min_distance = similarity_matrix.min(axis=0)
        # get the longest path
        longest_path = []
        longest_path.append(min_distance.argmin())
        for i in range(len(paths)-1, -1, -1):
            longest_path.append(paths[i][longest_path[-1]])
        return longest_path[::-1]

    def get_best_choice_by_interval(self, k: int=10) -> list[int]:
        """
        args:
            k: int, the number of intervals
        return:
            best_choice: list[int], the index of the best choice
        """
        # 对于每一个 masks, 我们考量其附件的 k=10 的 masks 的相似度
        # 我们选择最相似的作为最佳选择
        

def save_cache(var: any, filename: str):
    import pickle
    cache_dir = Path("test_cache")
    os.makedirs(cache_dir, exist_ok=True)
    file_path = cache_dir / filename
    with open(file_path, 'wb') as f:
        pickle.dump(var, f)
              
def get_cache(filename: str) -> any:
    import pickle
    cache_dir = Path("test_cache")
    file_path = cache_dir / filename
    with open(file_path, 'rb') as f:
        var = pickle.load(f)
    return var


def get_choice_list(masks_image_list: list[np.ndarray], scores_list: list[np.ndarray]) -> list[int]:
    not_empty_indices = [i for i, masks in enumerate(masks_image_list) if len(masks) > 0]
    not_empty_masks_image_list = [masks_image_list[i] for i in not_empty_indices]
    not_empty_scores_list = [scores_list[i] for i in not_empty_indices]
    s = SimilarityMatrix(not_empty_masks_image_list, not_empty_scores_list)
    choice_list = s.get_best_choices()
    # expand the choice list, the empty masks will be filled with -1
    choice_list = [choice_list.pop(0) if len(masks) > 0 else -1 for masks in masks_image_list]
    return choice_list

def get_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """get the bbox of the mask

    Args:
        mask (np.ndarray): [h, w]
    
    Returns:
        bbox (tuple[int, int, int, int]): (min_width, min_height, max_width, max_height)
    """
    from skimage import measure
    labels = measure.label(mask, connectivity=1)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda x: x.area, reverse=True)
    bbox = regions[0].bbox
    # this bbox is (min_height, min_width, max_height, max_width)
    # we need to convert it to (min_width, min_height, max_width, max_height)
    bbox = (bbox[1], bbox[0], bbox[3], bbox[2])
    return bbox


if __name__ == "__main__":
    masks_image_list = get_cache('masks_image.pkl')
    scores_list = get_cache('scores.pkl')
    s = SimilarityMatrix(masks_image_list, scores_list)
    l = s.get_best_choices()