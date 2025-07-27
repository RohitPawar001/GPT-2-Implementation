import tqdm
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.generate import generate_and_print_text, generate



def cal_loss_per_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch) 
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), target_batch.view(-1)
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_per_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss



def format_input(entry):

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately complates the request."
        f"\n\n## Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else " "

    )
    return instruction_text + input_text

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    input_batch, target_batch = zip(*batch)
    batch_max_length = max(item.shape[1] for item in input_batch)

    padded_inputs = []
    padded_targets = []
    
    for inputs, targets in zip(input_batch, target_batch):
        input = inputs.to(device)
        targets = targets.to(device)

        pad_length = batch_max_length - inputs.shape[1]
        if pad_length > 0:
            pad_tensor = torch.full((1, pad_length), pad_token_id, device=device)
            inputs = torch.cat([inputs, pad_tensor], dim=1)
            pad_targets = torch.full((1, pad_length), ignore_index, device=device)
            targets = torch.cat([targets, pad_targets], dim=1)

        if allowed_max_length is not None:
            inputs = inputs[:, :allowed_max_length]
            targets = targets[:, :allowed_max_length]

        padded_inputs.append(inputs)
        padded_targets.append(targets)

    input_tensors = torch.cat(padded_inputs, dim=0)
    target_tensors = torch.cat(padded_targets, dim=0)

    return input_tensors, target_tensors



class InstructionDataset(Dataset):

    def __init__(self, data, tokenizer, device="cpu"):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.encoded_text = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_text.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):

        encoded_text = self.encoded_text[index]
        # Return input and target tensors with correct shape and device
        inputs = torch.tensor(encoded_text[:-1], device=self.device).unsqueeze(0)
        targets = torch.tensor(encoded_text[1:], device=self.device).unsqueeze(0)
        return inputs, targets

    def __len__(self):
        return len(self.data)


def dataloaders(
    train_data,
    val_data,
    test_data,
    tokenizer,
    batch_size=2,
    device="cpu"
):
    train_dataset = InstructionDataset(train_data, tokenizer,device=device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    test_dataset = InstructionDataset(test_data, tokenizer,device=device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    val_dataset = InstructionDataset(val_data, tokenizer,device=device)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    return train_dataloader, test_dataloader, val_dataloader




def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer
    ):
    train_losses, val_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1

    accumulation_step = 4
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        for i, (input_batch, target_batch) in enumerate(pbar):
            loss = cal_loss_per_batch(
                input_batch, target_batch, model, device
            )

            loss = loss / accumulation_step
            loss.backward()

            if (i + 1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            token_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                torch.cuda.empty_cache()

                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(token_seen)
                pbar.set_postfix({
                    "train_loss": f"{train_loss:.3f}",
                    "val_loss": f"{val_loss:.3f}"
                })

        torch.cuda.empty_cache()

        print(f"\n[Epoch {epoch+1}] Sample Generation:")
        generate_and_print_text(
            model,
            tokenizer,
            device,
            start_context
        )
    print("Model training has been completed.")
    return train_losses, val_losses, track_token_seen


def finetune_model(
                 model,
                 train_loader,
                 val_loader,
                 optimizer,
                 device,
                 num_epochs,
                 tokenizer,
                 val_data,
                 eval_freqs=5,
                 eval_iter=5,
                 ):

         
   
    start_time = time.time()
    train_losses, val_losses, token_seen = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            eval_freq=eval_freqs,
            eval_iter=eval_iter,
            start_context=format_input(val_data[0]),
            tokenizer=tokenizer
        )

    end_time = time.time()
    execution_time_in_minutes = (end_time - start_time) / 60
    print(f"training completed in {execution_time_in_minutes:.2f} minutes.")
    return model
