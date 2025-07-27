import os
import torch

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '.'))


def save_model(model, model_name, optimizer=None):
    save_dir = os.path.join(parent_dir, "gpt_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    if optimizer is not None:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)
        print(f"Pretrained model has been saved successfully at {save_path}")
    else:
        save_path = os.path.join(parent_dir, "gpt_models", model_name)
        torch.save(model.state_dict(), save_path)
        print(f"Pretrained model has been saved successfully at {save_path}")





def load_model(model, model_name, device, model_args = None):

    save_dir = os.path.join(parent_dir, "gpt_models")
    model_path = os.path.join(save_dir, model_name)
    if model_args is not None:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model = model(model_args)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded pretrained model from {model_path}")
            return model
        else:
            raise FileNotFoundError("There is no pretrained model. You need to train the model first.")

    elif model_args == None:
        model.load_state_dict(torch.load(model_path))
        return model.to(device)



    