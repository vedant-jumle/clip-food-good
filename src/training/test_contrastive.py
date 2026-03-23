import torch
from torch.utils.data import Dataset, DataLoader

from src.models.clip_wrapper import CLIPWrapper
from src.training.trainer import train_contrastive


class DummyFoodDataset(Dataset):
    def __init__(self, num_samples=64, image_size=224, vocab_size=12):
        self.num_samples = num_samples
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.vocab = [f"ingredient_{i}" for i in range(vocab_size)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)  
        image = torch.randn(3, self.image_size, self.image_size)

        labels = torch.zeros(self.vocab_size, dtype=torch.float32)
        active_count = 2 + (idx % 3) 
        active_idx = torch.randperm(self.vocab_size)[:active_count]
        labels[active_idx] = 1.0

        return {
            "image": image,
            "labels": labels,
        }


class RepeatOneBatchLoader:
    def __init__(self, batch, repeats=20):
        self.batch = batch
        self.repeats = repeats

    def __iter__(self):
        for _ in range(self.repeats):
            yield self.batch

    def __len__(self):
        return self.repeats


def run_test(name, dataloader, vocab, epochs=2, lr=1e-4, rank=8, alpha=1.0, device="cpu"):
    print(f"\n=== {name} ===")
    clip = CLIPWrapper()

    losses = train_contrastive(
        clip=clip,
        dataloader=dataloader,
        vocab=vocab,
        epochs=epochs,
        lr=lr,
        rank=rank,
        alpha=alpha,
        device=device,
    )

    print("losses:", losses)
    assert len(losses) == epochs, f"Expected {epochs} losses, got {len(losses)}"
    assert all(torch.isfinite(torch.tensor(losses))), "Non-finite loss found"
    return losses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    dataset = DummyFoodDataset(num_samples=32, image_size=224, vocab_size=12)
    vocab = dataset.vocab

    smoke_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    smoke_losses = run_test(
        name="Smoke test",
        dataloader=smoke_loader,
        vocab=vocab,
        epochs=2,
        lr=1e-4,
        rank=8,
        alpha=1.0,
        device=device,
    )

    one_batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)))
    overfit_loader = RepeatOneBatchLoader(one_batch, repeats=10)
    overfit_losses = run_test(
        name="One-batch overfit test",
        dataloader=overfit_loader,
        vocab=vocab,
        epochs=3,
        lr=1e-4,
        rank=8,
        alpha=1.0,
        device=device,
    )

    print("\n=== Summary ===")
    print("smoke losses:", smoke_losses)
    print("overfit losses:", overfit_losses)

    if len(overfit_losses) >= 2 and overfit_losses[-1] <= overfit_losses[0]:
        print("Overfit test looks good: loss decreased.")
    else:
        print("Overfit test ran, but loss did not clearly decrease.")


if __name__ == "__main__":
    main()