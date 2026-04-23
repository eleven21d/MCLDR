#main.py
import torch
import time
import os
import csv
from datetime import datetime

import utility.batch_test
import utility.parser
import utility.tools
from utility.data_loader import Data
from model.model_light_gcrec import GCRec
from model.model import MCLDR
from utility.load_hete_data import load_data


def main():
    args = utility.parser.parse_args()
    device = torch.device(args.device)

    # Data loading
    dataset = Data(args.data_path + args.dataset, args)
    uu_graph, ii_graph = load_data(args.dataset, device)
    base_model = GCRec(args, dataset, uu_graph, ii_graph, device=device)

    # denoise path
    #denoise_path = "data/amazon_denoised0.02/G_denoised_beta0.02.txt"
    denoise_path = "data/mooc_v3denoised0.05/G_denoised_beta0.05.txt"
    #denoise_path = os.path.join(args.data_path, args.dataset, "G_denoised_beta0.05.txt")

    # Loaading MCLKR-Light
    mcldr_model = MCLDR(
        base_model=base_model,
        denoise_edge_path=denoise_path,
        denoise_embed_dim=int(args.dim),
        denoise_num_layers=2,
        denoise_lambda=float(args.denoise_lambda),
        denoise_temperature=float(args.temperature),
        device=device,
        fusion_alpha=float(args.fusion_alpha),
        replace_bpr=bool(args.replace_bpr)
    ).to(device)

    opt = torch.optim.Adam(mcldr_model.parameters(), lr=args.lr)
    best_recall, best_epoch = 0., 0
    best_result = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = "exp_results/mooc"
    ckpt_file = os.path.join(result_dir, f"{args.dataset}_best_{timestamp}.pth")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.dataset}_results_{timestamp}.csv")

    with open(result_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "dataset", "dim", "lr", "batch_size", "temperature",
            "ssl_lambda", "denoise_lambda", "epochs", "best_epoch","fusion_alpha",
            "Recall@20", "Recall@10", "Recall@5",
            "NDCG@20", "NDCG@10", "NDCG@5"
        ])

    print(f"MCLKR-Light training start on dataset: {args.dataset}")
    print(f"parser: lr={args.lr} | temp={args.temperature} | denoise_lambda={args.denoise_lambda} | batch={args.batch_size}")

    for epoch in range(args.epochs):
        start = time.time()
        mcldr_model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long().to(device)
        pos_items = torch.Tensor(sample_data[:, 1]).long().to(device)
        neg_items = torch.Tensor(sample_data[:, 2]).long().to(device)

        num_batch = len(users) // args.batch_size + 1
        total_loss_list = torch.zeros(5).to(device)

        for batch_u, batch_p, batch_n in utility.tools.mini_batch(
            users, pos_items, neg_items, batch_size=int(args.batch_size)
        ):
            loss_list = mcldr_model(batch_u, batch_p, batch_n)
            total_loss = sum(loss_list)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            for i, l in enumerate(loss_list):
                total_loss_list[i] += l.detach()

        avg_loss = (total_loss_list / num_batch).tolist()
        print(f"\tEpoch {epoch+1:03d} | Time: {time.time()-start:.2f}s "
              f"| BPR:{avg_loss[0]:.4f} | REG:{avg_loss[1]:.4f} "
              f"| SSL:{avg_loss[2]:.4f} | INTRA:{avg_loss[3]:.4f} | DENOISE:{avg_loss[4]:.4f}")

        if epoch % args.verbose == 0:
            result = utility.batch_test.Test(dataset, mcldr_model, device, args)
            recalls = result['recall']  # R@20, R@10, R@5
            ndcgs = result['ndcg']      # N@20, N@10, N@5

            print(f"\tRecall: {recalls} | NDCG: {ndcgs}")


            if recalls[0] > best_recall:
                best_recall = recalls[0]
                best_epoch = epoch + 1
                best_result = (recalls, ndcgs)
                torch.save(mcldr_model.state_dict(), ckpt_file)
                #torch.save(mcldr_model.state_dict(), ckpt_file)
                print(f"\t[Saved best ckpt] {ckpt_file}")

            '''
            if recalls[0] > best_recall:
                best_recall = recalls[0]
                best_epoch = epoch + 1
                best_result = (recalls, ndcgs)
            '''

    print(" Training completed.")
    print(f"Best epoch: {best_epoch} | Best Recall@20: {best_recall:.4f}")

    if best_result is not None:
        recalls, ndcgs = best_result
        row = [
            timestamp, args.dataset, args.dim, args.lr, args.batch_size,
            args.temperature, getattr(args, 'ssl_lambda', 'NA'), args.denoise_lambda,
            args.epochs, best_epoch,args.fusion_alpha,
            recalls[0], recalls[1], recalls[2],
            ndcgs[0], ndcgs[1], ndcgs[2]
        ]
        with open(result_file, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"result in: {result_file}")


if __name__ == '__main__':
    main()

