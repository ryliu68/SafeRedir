from torchvision.transforms import Resize
from torchvision.utils import save_image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_token_embedding(pipe, prompt):
    return pipe._encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )


@torch.no_grad()
def decode_latent_image(pipe, latent):
    latent = latent / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


ckpt_set = {
    "nudity": {
        "UCE": "nudity/UCE/UCE-Nudity-Diffusers-UNet.pt",
        "ESD": "nudity/ESD/ESD-Nudity-Diffusers-UNet-noxattn.pt",
        "SPM": "nudity/SPM/SPM-Nudity-Diffusers-UNet.pt",
        "FMN": "nudity/FMN/FMN-Nudity-Diffusers-UNet.pt",
        "AdvUnlearn": "nudity/AdvUnlearn/AdvUnlearn_Nudity_text_encoder_full.pt",
        "RECE": "nudity/RECE/nudity_ep2.pt",
        "MACE": "nudity/MACE",
        "ConcptPrune": "nudity/ConcptPrune/Nudity_skilled_neurons_0.01.pt",
        "DoCoPreG": "nudity/DoCoPreG/Nudity.bin",
        "Receler": "nudity/Receler"
    },
    "vangogh": {
        "UCE": "vangogh/UCE/UCE-VanGogh-Diffusers-UNet.pt",
        "ESD": "vangogh/ESD/ESD-VanGogh-Diffusers-UNet-xattn.pt",
        "SPM": "vangogh/SPM/SPM-VanGogh-Diffusers-UNet.pt",
        "FMN": "vangogh/FMN/FMN-VanGogh-Diffusers-UNet.pt",
        "AdvUnlearn": "vangogh/AdvUnlearn/AdvUnlearn_VanGogh_text_encoder_layer0.pt",
        "RECE": "vangogh/RECE/VanGogh_ep0.pt",
        "MACE": "vangogh/MACE",
        "ConcptPrune": "vangogh/ConcptPrune/VanGogh_skilled_neurons_0.01.pt",
        "DoCoPreG": "vangogh/DoCoPreG/Vangogh.bin",
        "Receler": "vangogh/Receler"
    },
    "object_church": {
        "AdvUnlearn": "object/AdvUnlearn/AdvUnlearn_Church_text_encoder_layer0.pt",  # 完成中
        "ESD": "object/ESD/ESD-Church-Diffusers-UNet-noxattn.pt",
        "FMN": "object/FMN/FMN-Church-Diffusers-UNet.pt",
        "SPM": "object/SPM/SPM-Church-Diffusers-UNet.pt",
        "RECE": "object/RECE/Church_ep0.pt",
        "MACE": "object/MACE/object_church",
        "UCE": "object/UCE/Church-sd_1_4.pt",
        "ConcptPrune": "object/ConcptPrune/Church_skilled_neurons_0.01.pt",
        "DoCoPreG": "object/DoCoPreG/Church.bin",
        "Receler": "object/Receler/Church"
    },
    "object_parachute": {
        "AdvUnlearn": "object/AdvUnlearn/AdvUnlearn_Parachute_text_encoder_layer0.pt",  # 完成
        "ESD": "object/ESD/ESD-Parachute-Diffusers-UNet-noxattn.pt",
        "FMN": "object/FMN/FMN-Parachute-Diffusers-UNet.pt",
        "SPM": "object/SPM/SPM-Parachute-Diffusers-UNet.pt",
        "RECE": "object/RECE/Parachute_ep0.pt",
        "MACE": "object/MACE/object_parachute",
        "UCE": "object/UCE/Parachute-sd_1_4.pt",
        "ConcptPrune": "object/ConcptPrune/Parachute_skilled_neurons_0.01.pt",
        "DoCoPreG": "object/DoCoPreG/Parachute.bin",
        "Receler": "object/Receler/Parachute"
    },
}


def resize(img_tensor, size=224):
    return Resize(size)(img_tensor)


class JointDatasetWithPairs(Dataset):
    def __init__(self, data_pth,train=True):
        self.data = torch.load(data_pth)
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.train:
            return (
                item["latent"].squeeze(0),
                item["prompt_embed"],
                torch.tensor([item["t"]], dtype=torch.float32),
                torch.tensor(item["label"], dtype=torch.long),
                item["safe_embed"],
                item["unsafe_embed"]
            )
        else:
            return (
                item["latent"].squeeze(0),
                item["prompt_embed"],
                torch.tensor([item["t"]], dtype=torch.float32),
                torch.tensor(item["label"], dtype=torch.long),
                item["unsafe_embed"]
            )


def get_dataloaders(data_pth, batch_size=64, num_workers=0, train=True):
    dataset = JointDatasetWithPairs(data_pth,train=train)
    split = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # train_loader,test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


