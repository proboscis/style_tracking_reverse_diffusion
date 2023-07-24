from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Tuple, List, Mapping, Callable

from deeplake import Dataset
from frozendict import frozendict

from torchvision.datasets import Places365
from tqdm import tqdm

from data_tree.util import Pickled
from proboscis_util.auto_image import AutoImage
from proboscis_util.rulebook import identify_image

@dataclass
class ImageData:
    """an image with metadata such as the src dataset and label"""
    image: AutoImage
    metadata: dict

    def __post_init__(self):
        self.metadata = frozendict(self.metadata)

    def __hash__(self):
        meta = frozendict(self.metadata)
        ary = self.image.to("numpy_rgb")
        return hash((meta, ary.tobytes()))

    def resize(self, size):
        return replace(self, image=self.image.resize_in_fmt(size))


class ImageDataset(ABC):
    @abstractmethod
    def __getitem__(self, item) -> Tuple[AutoImage, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sorted_unique_labels(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def label_to_indices(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def identifier(self) -> str:
        raise NotImplementedError

    def get_with_metadata(self, item) -> ImageData:
        img, label = self[item]
        return ImageData(img, {'label': label, 'idx': item, 'src': self.identifier})

    def sample_with_label_indices(self, indices: Mapping[str, List[int]], image_size):
        results = []
        for label, indices in tqdm(list(indices.items()), desc="sampling images"):
            for idx in indices:
                i = self.label_to_indices[label][idx]
                img = self.get_with_metadata(i)
                img.image = img.image.resize_in_fmt(image_size)
                results.append(img)
        return results

    def sample_with_indices(self, indices: List[int], image_size) -> List[ImageData]:
        results = []
        for idx in tqdm(indices, desc="sampling images"):
            img = self.get_with_metadata(idx)
            img.image = img.image.resize_in_fmt(image_size)
            results.append(img)
        return results

    def postprocess(self, processor: Callable[[AutoImage], AutoImage], identifier: str):
        return ProcessedImageDataset(self, identifier, processor)


@dataclass
class ProcessedImageDataset(ImageDataset):
    src: ImageDataset
    _identifier: str
    postprocess: Callable[[AutoImage], AutoImage]

    def __getitem__(self, item) -> Tuple[AutoImage, str]:
        img, label = self.src[item]
        return self.postprocess(img), label

    @property
    def sorted_unique_labels(self) -> List[str]:
        return self.src.sorted_unique_labels

    @property
    def label_to_indices(self) -> dict:
        return self.src.label_to_indices

    def __len__(self):
        return len(self.src)

    @property
    def identifier(self) -> str:
        return self._identifier


@dataclass
class MyPlaces365(ImageDataset):
    src: Places365
    index_cache_path: str

    def __post_init__(self):
        def _build_index_to_info():
            class_idx_to_images = defaultdict(list)
            class_idx_to_indices = defaultdict(list)
            for i in tqdm(range(len(self.src)), desc=f"building class to image index..."):
                img, cls = self.src[i]
                class_idx_to_images[cls].append(img)
                class_idx_to_indices[cls].append(i)
            return class_idx_to_images, class_idx_to_indices

        cache = Pickled(self.index_cache_path, _build_index_to_info)
        self.class_idx_to_images, self.class_idx_to_indices = cache.value
        self.class_idx_to_class = {i: self.src.classes[i] for i, c in enumerate(self.src.classes)}
        self.class_to_images = {self.class_idx_to_class[idx]: img for idx, img in self.class_idx_to_images.items()}
        self._sorted_unique_labels = list(sorted(self.class_to_images.keys()))
        self._class_to_indices = {self.class_idx_to_class[k]: v for k, v in self.class_idx_to_indices.items()}

    def __getitem__(self, item) -> Tuple[AutoImage, str]:
        img, cls = self.src[item]
        return identify_image(img), self.class_idx_to_class[cls]

    def __len__(self):
        return len(self.src)

    @property
    def sorted_unique_labels(self) -> List[str]:
        return self._sorted_unique_labels

    @property
    def label_to_indices(self) -> dict:
        return self._class_to_indices

    @property
    def identifier(self) -> str:
        return "places365"

# I think I need non cached version of wikiart.
@dataclass
class WikiartDS(ImageDataset):
    src: Dataset

    def __post_init__(self):
        # damn, for some reason this stopped working and I cant debug!
        # it seems the server is down
        self.images: "RemoteTensor" = self.src["images"]
        self.labels_in_integer = [int(i) for i in self.src["labels"].data()['value']]
        class_names = self.src["labels"].info['class_names']
        self.labels_in_string = [class_names[i] for i in self.labels_in_integer]
        self._label_to_indices = defaultdict(list)
        for idx, label_idx in enumerate(range(len(self.labels_in_integer))):
            self._label_to_indices[self.labels_in_string[idx]].append(idx)
            # wait.. the label is wrong?


    def __getitem__(self, item) -> Tuple[AutoImage, str]:
        return AutoImage.auto('numpy_rgb', self.images[item].data()['value']), self.labels_in_string[item]

    def __len__(self):
        return len(self.images)

    @property
    def sorted_unique_labels(self) -> List[str]:
        return list(sorted(self._label_to_indices.keys()))

    @property
    def label_to_indices(self) -> dict:
        return self._label_to_indices

    @property
    def identifier(self) -> str:
        return "wikiart"


