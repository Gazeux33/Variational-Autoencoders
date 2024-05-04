import os
import torch
import setup_data, engine, auto_encoder_model, encoder_model, decoder_model, utils, loss

from torchvision import transforms

saving = True

NUM_EPOCHS = 5
Z_DIM = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DATA_DIR = "../data"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader = setup_data.create_dataloaders(root=DATA_DIR,
                                                                  transform=data_transform,
                                                                  batch_size=BATCH_SIZE
                                                                  )

encoder_model = encoder_model.EncoderModelV2(z_dim=Z_DIM).to(device)
decoder_model = decoder_model.DecoderModelV0(z_dim=Z_DIM).to(device)

autoencoder = auto_encoder_model.VariationalAutoEncoderModelV0(encoder=encoder_model,
                                                               decoder=decoder_model,
                                                               kl_weight=100).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

loss_fn = loss.VaeLossV3()

if __name__ == "__main__":
    results = engine.train(model=autoencoder,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=NUM_EPOCHS,
                           device=device)
    print(results)
    utils.save_results(results, "results.json")
    if saving:
        utils.save_model(model=autoencoder,
                         target_dir="models",
                         model_name="05_going_modular_script_mode_tinyvgg_model.pth")
    utils.predict(autoencoder, test_dataloader)
    utils.plot_results(results)
