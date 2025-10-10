# @title Adding embeddings argument

from PIL import Image
import torch
from torch import nn
from torchvision import transforms, datasets
import argparse
from torch.utils.data import DataLoader
import os

from src.face_embedder.vision_transformer import vit_base

def get_model(pretrained_weights, device):
    input_size = 224
    model = vit_base(
        img_size=input_size,
        patch_size=14,
        init_values=1e-05,
        ffn_layer='mlp',
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)['teacher']
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    img_transforms = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, img_transforms

def get_embedding(model, imgs, device):
    imgs = imgs.to(device)
    with torch.no_grad():
        emb = model(imgs)
        emb = nn.functional.normalize(emb, p=2, dim=1)
    return emb

def main():
    parser = argparse.ArgumentParser(description="Compare or extract embeddings using a model.")
    parser.add_argument('--pretrained_weights', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--query_path', type=str, help='Path to query image')
    parser.add_argument('--gallery_path', type=str, required=True, help='Path to gallery image or folder')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). Auto-detect if not set.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top matches to show if gallery is a folder')
    parser.add_argument('--embeddings', type=str, help='Path to save embeddings (.pt file). If set, embeddings are generated and saved.')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_transforms = get_model(args.pretrained_weights, device)

    if args.embeddings:
        # Generate and save embeddings for the gallery
        if not os.path.isdir(args.gallery_path):
            raise ValueError("--embeddings requires gallery_path to be a folder")

        dataset = datasets.ImageFolder(args.gallery_path, transform=img_transforms)
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

        all_embs, all_paths = [], []
        for imgs, labels in dataloader:
            embs = get_embedding(model, imgs, device)
            all_embs.append(embs.cpu())
        all_embs = torch.cat(all_embs, dim=0)
        all_paths = [img_path for img_path, _ in dataset.imgs]

        torch.save({"embeddings": all_embs, "paths": all_paths}, args.embeddings)
        print(f"Saved {len(all_paths)} embeddings to {args.embeddings}")
        return

    # Default behavior (comparison)
    query_img = Image.open(args.query_path).convert('RGB')
    query_tensor = img_transforms(query_img).unsqueeze(0)
    emb_query = get_embedding(model, query_tensor, device)

    if os.path.isdir(args.gallery_path):
        dataset = datasets.ImageFolder(args.gallery_path, transform=img_transforms)
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

        gallery_embs = []
        for imgs, _ in dataloader:
            embs = get_embedding(model, imgs, device)
            gallery_embs.append(embs)
        gallery_embs = torch.cat(gallery_embs, dim=0)

        similarities = torch.cosine_similarity(emb_query.expand_as(gallery_embs), gallery_embs, dim=1)
        topk_vals, topk_indices = similarities.topk(min(args.top_k, similarities.size(0)), largest=True)
        gallery_paths = [img_path for img_path, _ in dataset.imgs]

        print(f"Top {topk_vals.size(0)} matches for query '{args.query_path}':")
        for i, (idx, sim) in enumerate(zip(topk_indices.tolist(), topk_vals.tolist()), 1):
            print(f"{i:2d}. {gallery_paths[idx]} | similarity: {sim:.4f}")
    else:
        gallery_img = Image.open(args.gallery_path).convert('RGB')
        gallery_tensor = img_transforms(gallery_img).unsqueeze(0)
        emb_gallery = get_embedding(model, gallery_tensor, device)
        similarity = torch.cosine_similarity(emb_query, emb_gallery).item()
        print(f"Cosine similarity between the query and the gallery image: {similarity:.4f}")

if __name__ == "__main__":
    main()

    # f.close()
