import argparse
import time
import yaml
import torch

from data.intention_dataloader import IntentionDataloader
from models.intention_predictor import IntentionPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark intention model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--gpu", type=int, default=0, help="Gpu id")
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--log-interval", type=int, default=50, help="interval of logging")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_config = config["DATA"]
    training_config = config["TRAINING"]
    model_config = config["MODEL"]

    if isinstance(data_config["resize"], int):
        data_config["resize"] = (data_config["resize"], data_config["resize"])

    data_config["batch_size"] = 1
    data = IntentionDataloader(**data_config)
    data.setup("fit")

    predictor = IntentionPredictor(
        data_config, training_config, data_len=len(data.train_dataloader()), **model_config
    )
    predictor.eval()
    predictor.to(f"cuda:{args.gpu}")

    torch.backends.cudnn.benchmark = False

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = 200

    data_loader = data.val_dataloader()
    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()

        with torch.no_grad():
            clip, boxes = data["clip"], data["boxes"]
            clip = clip.to(f"cuda:{args.gpu}")
            if boxes is not None:
                boxes = [boxes[0].to(f"cuda:{args.gpu}")]
            start_time = time.perf_counter()
            _ = predictor(clip, boxes)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f"Done image [{i + 1:<3}/ {total_iters}], " f"fps: {fps:.2f} img / s")

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f"Overall fps: {fps:.2f} img / s")
            break


if __name__ == "__main__":
    main()
