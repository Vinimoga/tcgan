import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pytorch_lightning.callbacks import Callback

class TSNECallback(Callback):
    def __init__(self, image_save_dir, latent_dim=100):
        self.image_save_dir = image_save_dir
        self.latent_dim = latent_dim

    def on_validation_end(self, trainer, pl_module):
        self._generate_and_save_tsne_plot(pl_module, trainer)

    def _generate_and_save_tsne_plot(self, pl_module, trainer):
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        with torch.no_grad():
            batch_size = 1000
            z = torch.randn(batch_size, self.latent_dim, device=pl_module.device)
            generated_images = pl_module.gen(z)
            generated_images = generated_images.view(generated_images.size(0), -1).cpu().numpy()

            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(generated_images)

            plt.figure(figsize=(8, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1, alpha=0.5)
            plt.title(f't-SNE of Generated Images - Epoch {trainer.current_epoch}')

            tsne_image_path = os.path.join(self.image_save_dir, f'tsne_epoch_{trainer.current_epoch}.png')
            plt.savefig(tsne_image_path)
            plt.close()

class KNNValidationCallback(Callback):
    def __init__(self, k=5):
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.datamodule.val_dataloader()
        for batch in val_loader:
            self.knn_validation(pl_module, batch)

    def knn_validation(self, pl_module, batch):
        x, y = batch
        pl_module.extract_and_load_encoder_from_discriminator()
        features = pl_module.encoder(x)

        self.knn.fit(features.cpu().detach().numpy(), y.cpu().numpy())
        predictions = self.knn.predict(features.cpu().detach().numpy())
        accuracy = accuracy_score(y.cpu().numpy(), predictions)

        pl_module.log("knn_accuracy", accuracy)
        print(f'KNN Validation Accuracy: {accuracy * 100:.2f}%')