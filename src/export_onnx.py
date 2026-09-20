import hydra
import onnx
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.export import Dim

from src.managers import DataManager
from src.models import PanguLightningModule, get_builder
from src.utils import DataCompose


@hydra.main(version_base=None, config_path="../config", config_name="export")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)

    eval_cases = list(set(sorted([datetime(2020, 1, 20)])))

    # prepare data
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(
        data_list, **cfg.data, **cfg.lightning, init_time_list=eval_cases
    )
    data_manager.setup("predict", quick=True)

    # sample data
    data_loader = data_manager.predict_dataloader()
    inp_data, oup_data = next(iter(data_loader))

    # model builder
    model_builder = get_builder(cfg.model.model_name)(
        "export_onnx",
        data_list,
        image_shape=data_manager.image_shape,
        add_time_features=cfg.data.add_time_features,
        **cfg.model,
        **cfg.lightning,
    )

    # load LightningModule from checkpoint
    pl_module = PanguLightningModule.load_from_checkpoint(
        checkpoint_path=cfg.inference.best_ckpt,
        test_dataloader=None,
        backbone_model=model_builder._backbone_model(),
        weights_only=False,
        image_shape=data_manager.image_shape,
        add_time_features=cfg.data.add_time_features,
        strict=False, # add the feature for _bin_idx and _S_count which are not saved in the checkpoint
        **cfg.model,
        **cfg.lightning,
    )

    # Overwrite landsea info inherited from training step
    # import torch
    # pl_module.backbone_model.patch_embed.topography_mask = torch.zeros((224,224)).to(pl_module.device)
    # pl_module.backbone_model.patch_embed.land_mask = torch.zeros((224,224)).to(pl_module.device)

    model_path:str = cfg.inference.best_ckpt

    model_name_parts = model_path.split("/")[-1].split("-")
    if len(model_name_parts) > 1:
        model_epoch = model_name_parts[1].split("=")[-1]
    else:
        model_epoch = "last"

    if hasattr(cfg.model, "model_label"):
        save_name = f"{model_name_parts[0]}_{cfg.model.model_label}_epoch_{model_epoch}"
    else:
        save_name = f"{model_name_parts[0]}_epoch_{model_epoch}"

    pl_module.eval()

    B = Dim("batch_size")

    # export onnx
    # pl_module = pl_module.cuda()
    # date = cfg.inference.best_ckpt.split("_")[1]  # e.g. 240831
    pl_module.to_onnx(
        file_path=f"./export/{save_name}.onnx",
        input_sample=(inp_data["upper_air"], inp_data["surface"]),
        verify=True,
        optimize=True,
        external_data=False,
        export_params=True,
        verbose=False,
        dynamic_shapes={
            "inputs": ({0: B}, {0: B}),
        },
    )

    print(f"Exported model to export/{save_name}.onnx")


def save_single_onnx():
    file_path = f"./export/Pangu_model_250611.onnx"
    model = onnx.load(file_path)
    onnx.save_model(
        model,
        "./export/Pangu_model_250611_101.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="Pangu_model_250611_101_external_data",  # same dir "./export/"
        size_threshold=10240,
        convert_attribute=False,
    )


if __name__ == "__main__":
    main()
