import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

def extract_embeddings(outputs):
    return outputs.hidden_states[0], outputs.hidden_states[-1]


def select_true_example(model, num_labels, data_frame, data_loader, tokenizer, device):
    correct_predictions = [[] for _ in range(num_labels)]
    sorted_input_embeddings = []
    sorted_output_embeddings = []
    sorted_attention_masks = []
    class_dfs = []
    batch_start_idx = 0
    for batch in tqdm(data_loader, desc="Evaluating model"):
        inputs, labels = batch
        attention_mask = inputs['attention_mask']
        inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            input_embeddings, output_embeddings = extract_embeddings(outputs)
        predictions = outputs.logits.argmax(dim=-1)

        for i in range(len(labels)):
            if predictions[i] == labels[i]:
                original_index = batch_start_idx + i
                correct_predictions[labels[i].item()].append((
                    outputs.logits[i, predictions[i]].item(),
                    data_frame.iloc[original_index].tolist(),
                    input_embeddings[i].cpu().numpy(),
                    output_embeddings[i].cpu().numpy(),
                    attention_mask[i].cpu().numpy().squeeze()  # Attention mask 저장
                ))
        batch_start_idx += len(labels)

    for i in range(num_labels):
        # logit 값에 따라 정렬
        correct_predictions[i].sort(key=lambda x: x[0], reverse=True)
        # 정렬된 데이터를 분리하여 저장
        sorted_class_data = [pred[1] for pred in correct_predictions[i]]
        sorted_input_embeds = [pred[2] for pred in correct_predictions[i]]
        sorted_output_embeds = [pred[3] for pred in correct_predictions[i]]
        sorted_attn_masks = [pred[4] for pred in correct_predictions[i]]  # Attention mask 정렬

        class_df = pd.DataFrame(sorted_class_data)
        class_dfs.append(class_df)

        sorted_input_embeddings.append(sorted_input_embeds)
        sorted_output_embeddings.append(sorted_output_embeds)
        sorted_attention_masks.append(sorted_attn_masks)  # Attention mask 저장

    return sorted_input_embeddings, sorted_attention_masks  # Attention mask도 반환


def extract_top_n_embeddings(extract_num, class_num, embeds_list, attn_masks_list):
    return_embeds_list = []
    return_masks_list = []

    for i in range(extract_num):
        extracted_embeds = []
        extracted_masks = []

        for j in range(class_num):
            # 임베딩 추출
            class_embeds = np.array(embeds_list[j][i:i + 1])
            class_embeds_tensor = torch.tensor(class_embeds)
            extracted_embeds.append(class_embeds_tensor)

            # 어텐션 마스크 추출
            class_attn_mask = np.array(attn_masks_list[j][i:i + 1])
            class_attn_mask_tensor = torch.tensor(class_attn_mask)
            extracted_masks.append(class_attn_mask_tensor)

        # 임베딩과 어텐션 마스크를 각각의 리스트에 추가
        return_embeds_list.append(extracted_embeds)
        return_masks_list.append(extracted_masks)

    return return_embeds_list, return_masks_list


def extract_bottom_n_embeddings(extract_num, class_num, embeds_list, attn_masks_list):
    return_embeds_list = []
    return_masks_list = []

    for i in range(extract_num):
        extracted_embeds = []
        extracted_masks = []

        for j in range(class_num):
            # 임베딩 추출
            class_embeds = np.array(embeds_list[j][-(i + 1):-(i)] or embeds_list[j][-(i + 1):])
            class_embeds_tensor = torch.tensor(class_embeds)
            extracted_embeds.append(class_embeds_tensor)

            # 어텐션 마스크 추출
            class_attn_mask = np.array(attn_masks_list[j][-(i + 1):-(i)] or attn_masks_list[j][-(i + 1):])
            class_attn_mask_tensor = torch.tensor(class_attn_mask)
            extracted_masks.append(class_attn_mask_tensor)

        # 임베딩과 어텐션 마스크를 각각의 리스트에 추가
        return_embeds_list.append(extracted_embeds)
        return_masks_list.append(extracted_masks)

    return return_embeds_list, return_masks_list

def get_decimal_precision(value):
    str_value = str(value)
    if '.' in str_value:
        return len(str_value.split('.')[1])
    else:
        return 0

def perturb(embeds, eps, grad):
    perturbed_embeds = embeds - eps*grad.sign()
    return perturbed_embeds

def calculate_gradient(model, input_embeds, target_class, device):
    # print(input_embeds.shape)
    input_embeds = input_embeds.clone().detach().requires_grad_(True).to(device)
    target = torch.tensor([target_class]).to(device)
    outputs = model(inputs_embeds=input_embeds, labels=target)
    init_pred = outputs.logits.argmax(dim=-1)
    loss = outputs.loss
    model.zero_grad()
    loss.backward()
    data_grad = input_embeds.grad
    return loss, data_grad


def fgsm_attack(model, source, target, input_embeds, start_eps, eps_step, max_eps, device):
    input_embeds = input_embeds.to(device)
    digits = get_decimal_precision(eps_step)
    targeted_eps = start_eps
    loss, data_grad = calculate_gradient(model, input_embeds, target, device)
    first_diff_eps = math.inf

    while targeted_eps <= max_eps:
        # print(f"epsilon : {eps}")
        perturbed_embeds = perturb(input_embeds, targeted_eps, data_grad)
        adv_outputs = model(inputs_embeds=perturbed_embeds)
        adv_pred = adv_outputs.logits.argmax(dim=-1)
        # print(adv_pred)

        if first_diff_eps == math.inf and adv_pred.item() != source:
            first_diff_eps = targeted_eps

        if adv_pred.item() == target:
            # print(f"{eps} adv attack success!")
            break
        else:
            targeted_eps += eps_step
            targeted_eps = round(targeted_eps, digits)
    return targeted_eps, first_diff_eps

def calculate_all_epsilon(model, class_num, logit_example, device, step_eps=0.01, max_eps=10.0):
    first_changed_list = []
    targeted_list = []
    for i in range(0, class_num):
        # print(f"class {i}")
        targeted_temp = []
        first_changed_temp = []
        for j in range(0, class_num):
            if i == j:
                targeted_temp.append((i, j, math.inf))
                first_changed_temp.append((i, j, math.inf))
                continue
            eps, first_diff_eps = fgsm_attack(model, i, j, logit_example[i], 0.00, step_eps, max_eps, device)
            if eps >= max_eps:
                targeted_temp.append((i, j, math.inf))
            else:
                targeted_temp.append((i, j, eps))
            first_changed_temp.append((i, j, first_diff_eps))
        targeted_list.append(targeted_temp)
        first_changed_list.append(first_changed_temp)

    flat_targeted_list = [(source, target, eps) for sublist in targeted_list for source, target, eps in sublist if
                          eps != math.inf]
    flat_first_changed_list = [(source, target, eps) for sublist in first_changed_list for source, target, eps in
                               sublist if eps != math.inf and eps - (2 * step_eps) >= 0.00]

    return flat_targeted_list, flat_first_changed_list

def generate_example(model, device, source, target, input_embeds, example_num, boundary_eps, step_eps):
    example_list = []
    example_label = []
    source_embedding = input_embeds[source].to(device)
    # Gradient 계산
    loss, data_grad = calculate_gradient(model, source_embedding, target, device)
    iter_num = 0
    while iter_num < example_num:
        # Perturbation 적용
        eps = random.uniform(boundary_eps, boundary_eps + step_eps)
        generated_embeds = perturb(source_embedding, eps, data_grad)
        example_list.append(generated_embeds.cpu())  # CPU로 이동
        example_label.append(source)
        iter_num += 1
        del generated_embeds
        torch.cuda.empty_cache()  # GPU 메모리 캐시 비우기
    del source_embedding, loss, data_grad
    torch.cuda.empty_cache()  # 캐시 비우기
    return example_label, example_list

def adjust_examples(example_list, example_label, eps_list, embed_list, attn_mask_list, target_num, step_eps, generate_fn, model, device):
    if len(example_list) < target_num:
        diff = target_num - len(example_list)
        for _ in range(diff):
            random_index = random.randint(0, len(embed_list)-1)
            input_embed = embed_list[random_index]
            eps_values = eps_list[random_index]
            attn_mask = attn_mask_list[random_index]
            if eps_values:
                source, target, eps = random.choice(eps_values)
                label, example = generate_fn(model, device, source, target, input_embed, 1, eps-2*step_eps, step_eps)
                example_label.extend(label)
                example_list.extend(example)
                attn_mask_list.extend(attn_mask.unsqueeze(0))
    elif len(example_list) > target_num:
        diff = len(example_list) - target_num
        for _ in range(diff):
            random_index = random.randint(0, len(example_list) - 1)
            example_list.pop(random_index)
            example_label.pop(random_index)
            attn_mask_list.pop(random_index)


def make_example(model, data_frame, data_loader, tokenizer, example_num, top_emb, bottom_emb, class_num, true_ratio,
                 device, step_eps=0.01, max_eps=10.0):
    extract_embed_list = []
    extract_attn_list = []
    negative_eps = []
    positive_eps = []
    positive_attn_mask_list = []
    positive_example_list = []
    positive_example_label = []
    negative_example_list = []
    negative_example_label = []
    negative_attn_mask_list = []

    positive_num = int(round(example_num * true_ratio * 0.01))
    negative_num = example_num - positive_num
    extract_num = top_emb + bottom_emb

    print("Generate positive num : ", positive_num)
    print("Generate negative num : ", negative_num)

    per_emb_positive_example_num = int(round(positive_num / extract_num))
    per_emb_negative_example_num = int(round(negative_num / extract_num))

    print("Extract Num : ", extract_num)
    print("Generate per embedding positive example num : ", per_emb_positive_example_num)
    print("Generate per embedding negative example num : ", per_emb_negative_example_num)

    input_embeds, attn_masks = select_true_example(model, class_num, data_frame, data_loader, tokenizer, device)

    # extract_embed_list.extend(extract_top_n_embeddings(top_emb, class_num, input_embeds))
    # extract_embed_list.extend(extract_bottom_n_embeddings(bottom_emb, class_num, input_embeds))

    extract_embed, extract_attn = extract_top_n_embeddings(top_emb, class_num, input_embeds, attn_masks)
    extract_embed_list.extend(extract_embed)
    extract_attn_list.extend(extract_attn)

    extract_embed, extract_attn = extract_bottom_n_embeddings(bottom_emb, class_num, input_embeds, attn_masks)
    extract_embed_list.extend(extract_embed)
    extract_attn_list.extend(extract_attn)

    for i in range(0, len(extract_embed_list)):
        targeted_eps, first_changed_eps = calculate_all_epsilon(model, class_num, extract_embed_list[i], device, step_eps,
                                                                max_eps)
        positive_eps.append(first_changed_eps)
        negative_eps.append(targeted_eps)

    for i, (input_embed, attn_mask) in enumerate(zip(extract_embed_list, extract_attn_list)):
        for j, first_changed in enumerate(positive_eps[i]):
            per_positive_example_num = int(round(per_emb_positive_example_num / len(positive_eps[i])))
            source, target, eps = first_changed
            # print("per positive example num", per_positive_example_num)
            pos_label, pos_example = generate_example(model, device, source, target, extract_embed_list[i],
                                                      per_positive_example_num, eps - 2 * step_eps, step_eps)
            positive_example_label.extend(pos_label)
            positive_example_list.extend(pos_example)
            for k in range(0, per_positive_example_num):
                positive_attn_mask_list.extend(attn_mask[source].unsqueeze(0))

        for j, targeted in enumerate(negative_eps[i]):
            per_negative_example_num = int(round(per_emb_negative_example_num / len(negative_eps[i])))
            source, target, eps = targeted
            # print("per negative example num", per_negative_example_num)
            neg_label, neg_example = generate_example(model, device, source, target, extract_embed_list[i],
                                                      per_negative_example_num, eps, step_eps)
            negative_example_label.extend(neg_label)
            negative_example_list.extend(neg_example)
            for k in range(0, per_negative_example_num):
                negative_attn_mask_list.extend(attn_mask[source].unsqueeze(0))

    adjust_examples(positive_example_list, positive_example_label, positive_eps, extract_embed_list,
                    positive_attn_mask_list, positive_num, step_eps, generate_example, model, device)
    adjust_examples(negative_example_list, negative_example_label, negative_eps, extract_embed_list,
                    negative_attn_mask_list, negative_num, step_eps, generate_example, model, device)
    return positive_example_label, positive_example_list, positive_attn_mask_list, negative_example_label, negative_example_list, negative_attn_mask_list