import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.data_utils import *
from utils.example_gen import *
from utils.model_utils import *
from utils.visualize import *
import os

def main():
    parser = argparse.ArgumentParser(description='Generate examples using the provided model and dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model directory')
    parser.add_argument('--data', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--example_num', type=int, required=True, help='Number of examples to generate')
    parser.add_argument('--top_emb', type=int, required=True, help='Number of top embeddings to use')
    parser.add_argument('--bottom_emb', type=int, required=True, help='Number of bottom embeddings to use')
    parser.add_argument('--class_num', type=int, required=True, help='Number of classes')
    parser.add_argument('--true_ratio', type=int, default=90, help='Ratio of true examples')
    parser.add_argument('--step_eps', type=float, default=0.01, help='Step epsilon for FGSM')
    parser.add_argument('--max_eps', type=float, default=5.0, help='Max epsilon for FGSM')

    args = parser.parse_args()

    # Load model and tokenizer
    device = set_device()
    model, tokenizer = load_model(args.model_path, device)

    # 결과 저장 경로 설정
    model_name = os.path.basename(args.model_path)  # model_path에서 마지막 디렉토리/파일 이름 추출
    result_dir = os.path.join("result", model_name)  # 'result' 디렉토리 아래에 모델 이름으로 폴더 생성
    os.makedirs(result_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # Load validation data
    df = pd.read_csv(args.data, header=None, names=["class", "question_title", "question_body", "answer"])
    dataset = TestDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Generate examples
    positive_example_label, positive_example_list, positive_attn_list, negative_example_label, negative_example_list, negative_attn_list = make_example(
        model=model,
        data_frame=df,
        data_loader=loader,
        tokenizer=tokenizer,
        example_num=args.example_num,
        top_emb=args.top_emb,
        bottom_emb=args.bottom_emb,
        class_num=args.class_num,
        true_ratio=args.true_ratio,
        device=device,
        step_eps=args.step_eps,
        max_eps=args.max_eps
    )

    print("Positive Examples Generated:", len(positive_example_list))
    print("Negative Examples Generated:", len(negative_example_list))

    print(ret_classification_report(model, positive_example_list+negative_example_list, positive_example_label+negative_example_label, device))
    save_result(
        model=model,
        extract_num=1000,
        p_list=positive_example_list,
        n_list=negative_example_list,
        p_attn=positive_attn_list,
        n_attn=negative_attn_list,
        df=df,
        class_num=args.class_num,
        loader=loader,
        tokenizer=tokenizer,
        result_dir=result_dir,
        device=device
    )


if __name__ == "__main__":
    main()