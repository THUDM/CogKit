import argparse
import yaml

from cogkit.finetune import get_model_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True)
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        config = yaml.safe_load(f)

    trainer_cls = get_model_cls(
        config["model_name"], config["training_type"], config["enable_packing"]
    )
    trainer = trainer_cls(args.yaml)
    trainer.fit()


if __name__ == "__main__":
    main()
