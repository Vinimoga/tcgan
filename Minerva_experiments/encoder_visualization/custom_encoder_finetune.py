from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from minerva.models.ssl.cpc import CPC
from minerva.models.nets.cpc_networks import HARCPCAutoregressive
from minerva.data.data_modules.har_rodrigues_24 import HARDataModuleCPC
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.models.nets.base import SimpleSupervisedModel

from minerva.models.nets.tnc import TSEncoder
import torchmetrics

from minerva.data.data_modules.har import MultiModalHARSeriesDataModule
from minerva.models.loaders import FromPretrained
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
from minerva.analysis.metrics.balanced_accuracy import BalancedAccuracy
from minerva.analysis.model_analysis import TSNEAnalysis

from minerva.models.nets.time_series.gans import GAN

import sys
import os

from TCGAN_model import TCGAN_Discriminator, TCGAN_Generator, TCGAN_Encoder

import psutil
import os

p = psutil.Process(os.getpid())
p.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])  # Limita o processo a rodar apenas nos núcleos 0 e 1


epochs = 100
data_dir = '../../../standardize_view' # '../../../standardize_view' se estiver utilizando run e 'Projetos/standardize_view' utilizando debug
datasets = os.listdir(data_dir)
batch_size = 64
root_ckpt_dir = '../trained_models' #'../trained_models' se estiver utilizando run "Projetos/tcgan/Minerva_experiments/trained_models" utilizando debug
for current_iteraction in range(1, 3): 

    ckpt_path_list = (os.listdir(f"{root_ckpt_dir}/{current_iteraction}"))
    ckpt_path_list.sort()

    for dataset_current_number in range(6):
        for ckpt_current_number in range(6):
            experiment_name = f'new/experiment_{ckpt_current_number + 1}'

        
            print("\n\n--------------------------------------------------")
            print(f"Dataset {datasets[dataset_current_number]}, Experiment {ckpt_current_number + 1}, Iteration {current_iteraction}")
            print("------------------------------------------------------")

            # Name of the experiment
            execution_id = f'run_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
            # Directory to save logs
            log_dir = f"./logs/{experiment_name}/{datasets[dataset_current_number]}/{execution_id}" 


            print(f"Execution ID: {execution_id}")
            print(f"Log dir: {log_dir}\n\n")


            data_module = MultiModalHARSeriesDataModule(
                data_path=f"{data_dir}/{datasets[dataset_current_number]}",
                feature_prefixes=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                label="standard activity code",
                features_as_channels=True,
                cast_to="float32",
                batch_size=batch_size,
                num_workers=4,
            )

            # REPLACE THIS WITH YOUR ENCODER CHECKPOINT PATH
            ckpt_path = f'{root_ckpt_dir}/{current_iteraction}/{ckpt_path_list[ckpt_current_number]}/model.pth'
            print(f'Current checkpoint path: {ckpt_path}\n')
            ckpt = torch.load(f=ckpt_path)
            print(f'original keys: {ckpt["dis_state_dict"].keys()}\n')

            # Load model and give it the state_dict
            model = TCGAN_Discriminator()  
            model.load_state_dict(ckpt['dis_state_dict'], strict=False)


            print(model.state_dict()['l1.weight'][0][0][0],
                model.state_dict()['backbone.0.weight'][0][0][0], '\n') 
                #Checking if model have the correct shape and if it have been loaded succefully

            # Instantiating encoder (random weights)
            backbone = TCGAN_Encoder()

            backbone = FromPretrained(
                model=backbone,
                ckpt_path=ckpt_path,
                strict=False,
                ckpt_key='dis_state_dict'
            )
            print('l7.weight and l7.bias are the Head of the discriminator Model\n')
            # Pega os dataloader de treino
            data_module.setup("fit")
            train_data_loader = data_module.train_dataloader()
            for batch in train_data_loader:
                X, y = batch
                #print(f"X shape: {X.shape}, y shape: {y.shape}")
                break

            #print(f"O primeiro batch de treino tem shape X={tuple(X.shape)} e y={tuple(y.shape)}\n")


            embeddings = backbone(X)
            mlp_input_shape = embeddings.shape[1]
            #print(f"O embedding tem shape {tuple(embeddings.shape)}")

            num_classes = 6
            head = MLP([mlp_input_shape, 128, num_classes])


            model = SimpleSupervisedModel(
                backbone=backbone,
                fc=head,
                loss_fn=torch.nn.CrossEntropyLoss(),
                flatten=False,
                train_metrics={
                    "acc": torchmetrics.Accuracy(task="multiclass", num_classes=6),
                },
                val_metrics={
                    "acc": torchmetrics.Accuracy(task="multiclass", num_classes=6),
                },
            )

            ## Callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints/',
                monitor='val_loss',
                mode='min',
                save_last=True
            )

            ## Logger
            logger = CSVLogger(save_dir=log_dir, name='tc-finetune', version=execution_id)

            ## Trainer
            trainer = L.Trainer(
                # Maximum number of epochs to train
                max_epochs=epochs,
                # Training on GPU
                accelerator="gpu",
                # We will train using 1 gpu
                devices=1,
                # Logger for logging
                logger=logger,
                # List of callbacks
                callbacks=[checkpoint_callback],
                # Only for testing. Remove for production. We will only train using 1 batch of training and validation
                #limit_train_batches=1,
                #limit_val_batches=1,
                enable_progress_bar=False,
            )

            train_pipeline = SimpleLightningPipeline(
                model=model,
                trainer=trainer,
                log_dir=log_dir,
                save_run_status=True,
                seed=42
            )

            train_pipeline.run(data_module, task="fit")

            test_pipeline = SimpleLightningPipeline(
                model=model,
                trainer=trainer,
                log_dir=log_dir,
                save_run_status=True,
                seed=42,
                classification_metrics={
                    "accuracy": torchmetrics.Accuracy(num_classes=6, task="multiclass"),
                    "f1": torchmetrics.F1Score(num_classes=6, task="multiclass"),
                    "precision": torchmetrics.Precision(num_classes=6, task="multiclass"),
                    "recall": torchmetrics.Recall(num_classes=6, task="multiclass"),
                    "balanced_accuracy": BalancedAccuracy(num_classes=6, task="multiclass"),
                },
                apply_metrics_per_sample=False,
                model_analysis={
                    "tsne": TSNEAnalysis(
                        height=800,
                        width=800,
                        legend_title="Activity",
                        title="t-SNE of CPC Finetuned on KuHar",
                        output_filename="tsne_cpc_finetuned_kuhar.pdf",
                        label_names={
                            0: "sit",
                            1: "stand",
                            2: "walk",
                            3: "stair up",
                            4: "stair down",
                            5: "run",
                            6: "stair up and down",
                        },
                    )
                },
            )

            d = test_pipeline.run(
                data_module, task="evaluate", ckpt_path=checkpoint_callback.best_model_path
            )
            print(d)