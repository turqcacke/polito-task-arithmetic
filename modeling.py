import open_clip
import torch
import utils
from typing import Mapping, Optional, Any, Union, List, NamedTuple
from args import ArgsProto

class StateDictMising(NamedTuple):
    missing_keys: List[Any]
    unexpected_keys: List[Any]


class ImageEncoder(torch.nn.Module):
    def __init__(self, args: ArgsProto, keep_lang=False) -> None:
        super().__init__()
        name, pretrained = self.extract_model_args(args)
        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images: Any) -> Union[torch.Tensor, Any]:
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs: Any) -> Union[torch.Tensor, Any]:
        return self.forward(inputs)

    def save(self, filename: str) -> None:
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, args: ArgsProto, filename: str) -> StateDictMising:
        print(f"Loading image encoder from {filename}")
        state_dict: Mapping[str, Any] = torch.load(filename, map_location="cpu")
        return cls.load_from_state_dict(args, state_dict)

    @classmethod
    def load_from_state_dict(cls, args: ArgsProto, state_dict: Mapping[str, Any]) -> StateDictMising:
        model, pretrained = cls.extract_model_args(args)
        (
            cls.model,
            cls.train_preprocess,
            cls.val_preprocess,
        ) = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, cache_dir=args.openclip_cachedir
        )
        return cls.model.load_state_dict(state_dict)
    
    @classmethod
    def extract_model_args(cls, args: ArgsProto) -> tuple[str, Optional[str]]:
        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"
        return name, pretrained

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs: Any) -> torch.Tensor:
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs: Any) -> torch.Tensor:
        return self.forward(inputs)

    def save(self, filename: str) -> None:
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename: str) -> Any:
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder: ImageEncoder, classification_head: ClassificationHead):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self) -> None:
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs: Any) -> torch.Tensor:
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs: Any) -> torch.Tensor:
        return self.forward(inputs)

    def save(self, filename: str) -> None:
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename: str) -> Any:
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)