
import torch


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None
):
    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.block_size:]

        logits, _ = model(context)

        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )
            logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities, num_samples=1
        )

        input_ids = torch.cat(
            (input_ids, next_token), dim=1
        )

    return input_ids
