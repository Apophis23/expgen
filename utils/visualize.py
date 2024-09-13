import torch
import numpy as np
import random
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.example_gen import *

def ret_classification_report(model, emb_list, label_list, device):
    predictions = []
    with torch.no_grad():
        for i, input_embeds in tqdm(enumerate(emb_list), total=len(emb_list), desc="Evaluating"):
            # print(input_embeds.shape)
            outputs = model(inputs_embeds=input_embeds.to(device))
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
        report = classification_report(np.array(label_list), predictions)
    return report


def extract_cls_token(model, embedding_list, attn_masks_list, device):
    cls_token_list = []
    preds_list = []

    for input_embeds, attn_mask in zip(embedding_list, attn_masks_list):
        input_embeds = input_embeds.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            outputs = model(inputs_embeds=input_embeds, attention_mask=attn_mask, output_hidden_states=True)
        preds = outputs.logits.argmax(dim=-1)

        cls_token_embed = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        cls_token_list.extend(cls_token_embed)
        preds_list.append(preds.item())

    return np.array(cls_token_list), preds_list


def select_random_samples(token_list, num_samples=10):
    return random.sample(token_list, num_samples)

# TSNE 시각화 및 저장
def visualize_embeddings_TSNE(original_cls_token_list, predict_token_list, class_num, result_dir, file_name, random_state=42):
    tsne = TSNE(n_components=2, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(original_cls_token_list)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=predict_token_list, cmap='tab10',
                          edgecolor='k', s=40)

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_labels = [f"class {i + 1}" for i in range(class_num)]
    legend = plt.legend(handles, legend_labels, title="Classes")

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # 저장
    plt.savefig(f"{result_dir}/tsne_{file_name}.png")  # 전달받은 result_dir 사용
    plt.show()

# PCA 시각화 및 저장
def visualize_embeddings_PCA(original_cls_token_list, predict_token_list, class_num, result_dir, file_name):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(original_cls_token_list)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=predict_token_list, cmap='tab10',
                          edgecolor='k', s=40)

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_labels = [f"class {i + 1}" for i in range(class_num)]
    legend = plt.legend(handles, legend_labels, title="Classes")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # 저장
    plt.savefig(f"{result_dir}/pca-{file_name}.png")  # 전달받은 result_dir 사용
    plt.show()

def save_result(model, extract_num, p_list, n_list, p_attn, n_attn, df, class_num, loader, tokenizer, device, result_dir):
    input_embeds, att_masks = select_true_example(model, class_num, df, loader, tokenizer, device)

    plot_list = []
    plot_mask = []
    value, mask = extract_top_n_embeddings(extract_num, class_num, input_embeds, att_masks)
    plot_list.extend(value)
    plot_mask.extend(mask)

    exp = []
    msk = []
    for i in range(1000):
        for j in range(10):
            exp.append(plot_list[i][j])
            msk.append(plot_mask[i][j])

    cls_list, preds_list = extract_cls_token(model, exp, msk, device)
    visualize_embeddings_TSNE(cls_list, preds_list, 10, result_dir, "original_data", random_state=42)  # result_dir 전달
    visualize_embeddings_PCA(cls_list, preds_list, 10, result_dir, "original_data")  # result_dir 전달

    all_list = exp + p_list + n_list
    all_msk = msk + p_attn + n_attn
    cls_list, preds_list = extract_cls_token(model, all_list, all_msk, device)
    visualize_embeddings_TSNE(cls_list, preds_list, 10, result_dir, "generated_data", random_state=42)  # result_dir 전달
    visualize_embeddings_PCA(cls_list, preds_list, 10, result_dir, "generated_data")  # result_dir 전달
