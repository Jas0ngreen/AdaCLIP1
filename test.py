import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import torch
from scipy.ndimage import gaussian_filter
import cv2

# Importing from local modules
from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method import AdaCLIP_Trainer
from PIL import Image
import numpy as np

setup_seed(111)

def train(args):
    assert os.path.isfile(args.ckt_path), f"Please check the path of pre-trained model, {args.ckt_path} is not valid."
    compare_shapes = getattr(args, 'compare_loss_shapes', False)
    reference_path = getattr(args, 'source_reference_report', None)
    if compare_shapes and (not args.diagnose_source_losses or not reference_path):
        raise ValueError("--compare_loss_shapes requires --diagnose_source_losses and --source_reference_report")
    if reference_path and not compare_shapes:
        raise ValueError("--source_reference_report requires --compare_loss_shapes")
    reference = None
    diagnostic_name = 'source_gate_loss_shapes' if compare_shapes else 'source_gate_losses'
    if args.diagnose_source_losses:
        if (not args.load_baseline or args.testing_model != "dataset" or args.prompting_type != "SD"
                or args.check_gate_equivalence or args.diagnose_prompt_scale or args.gate_override is not None):
            raise ValueError("Source diagnosis requires --load_baseline, SD, dataset mode and no other diagnostic/gate override")
        if args.source_samples_per_group < 1 or len(args.source_data) != len(set(args.source_data)):
            raise ValueError("Use positive source_samples_per_group and distinct source datasets")
        report_path = os.path.join(args.save_path, 'diagnostics', diagnostic_name + '.json')
        if os.path.exists(report_path):
            raise FileExistsError(f"Report exists; use a new --save_path: {report_path}")
        if compare_shapes:
            with open(reference_path, 'r') as handle:
                reference = json.load(handle)
    if args.load_baseline:
        if args.fusion_mode != "shared_gate":
            raise ValueError("--load_baseline requires --fusion_mode shared_gate")
        if not (args.check_gate_equivalence or args.diagnose_prompt_scale or args.diagnose_source_losses) and args.gate_override is None:
            raise ValueError("An imported baseline has no learned gate; specify --gate_override")
    if args.diagnose_prompt_scale:
        if not args.load_baseline or args.testing_model != "dataset" or args.check_gate_equivalence:
            raise ValueError("--diagnose_prompt_scale requires --load_baseline, dataset mode, and no equivalence flag")
        if args.gate_override is not None or args.prompting_type != "SD" or args.equivalence_per_group < 1:
            raise ValueError("Prompt diagnosis uses gates 1/0.5 internally; require SD, positive per-group count, no --gate_override")
    if args.check_gate_equivalence:
        if not args.load_baseline or args.testing_model != "dataset":
            raise ValueError("--check_gate_equivalence requires --load_baseline and dataset testing")
        if args.gate_override not in (None, 1.0):
            raise ValueError("Equivalence checking compares add with gate=1 only")
        if args.equivalence_per_group < 1:
            raise ValueError("--equivalence_per_group must be positive")

    # Configurations
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_fig = args.save_fig

    # Logger
    if args.diagnose_source_losses:
        os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
        logger = Logger(os.path.join(args.save_path, 'logs', diagnostic_name + '.txt'))
    elif args.load_baseline:
        os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
        if args.diagnose_prompt_scale:
            label = 'prompt_scale'
        else:
            label = 'equivalence' if args.check_gate_equivalence else f'fixed_{args.gate_override:g}'
        logger = Logger(os.path.join(args.save_path, 'logs', f'{args.testing_data}_baseline_shared_gate_{label}.txt'))
    else:
        logger = Logger('log.txt')

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')


    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # Prepare model
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
        fusion_mode=args.fusion_mode,
    ).to(device)

    if args.load_baseline:
        count = model.load_baseline_for_shared_gate(args.ckt_path)
        logger.info(f'Baseline import PASS: {count} detector tensors copied exactly; gate is untrained.')
    else:
        model.load(args.ckt_path)
    if args.gate_override is not None:
        if args.fusion_mode != "shared_gate" or not 0 <= args.gate_override <= 1:
            raise ValueError("--gate_override requires shared_gate and a value in [0, 1]")

    if args.diagnose_source_losses:
        from tools.source_loss_diagnostics import select_source_samples, diagnose_source_losses
        logger.info('SOURCE-ONLY diagnostic: --testing_data is ignored; no target dataset will be loaded.')
        source_loaders, sampling = [], {}
        for source in args.source_data:
            _, source_dataset, source_root = get_data(
                dataset_type_list=source, transform=model.preprocess,
                target_transform=model.transform, training=False)
            indices, coverage = select_source_samples(source_dataset, args.source_samples_per_group, args.source_sample_seed)
            sampling[source] = {"root": source_root, "coverage": coverage}
            subset = torch.utils.data.Subset(source_dataset, indices)
            source_loaders.append((source, torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)))
            logger.info(f'Source={source}: sampled={len(indices)}, augmentation=off, metadata_population=test')
        metadata = {"checkpoint": os.path.abspath(args.ckt_path), "model": args.model,
                    "sources": args.source_data, "sample_seed": args.source_sample_seed,
                    "samples_per_group": args.source_samples_per_group, "sampling": sampling,
                    "image_size": args.image_size, "use_hsf": args.use_hsf, "k_clusters": args.k_clusters}
        diagnose_source_losses(model, source_loaders, logger, report_path, metadata,
                               compare_shapes=compare_shapes, reference=reference)
        return

    if args.testing_model == 'dataset':
        assert args.testing_data in dataset_dict.keys(), f"You entered {args.testing_data}, but we only support " \
                                                         f"{dataset_dict.keys()}"

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')
        suffix = ''
        if args.fusion_mode == "shared_gate":
            gate_label = (
                "learned" if args.gate_override is None
                else f"fixed_{args.gate_override:g}"
            )
            suffix = f'_shared_gate_{gate_label}'
            if args.load_baseline:
                suffix = '_baseline' + suffix
        csv_path = os.path.join(csv_root, f'{args.testing_data}{suffix}.csv')
        image_dir = os.path.join(image_root, f'{args.testing_data}{suffix}')
        test_data_cls_names, test_data, test_data_root = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False)

        if args.diagnose_prompt_scale:
            from tools.gate_equivalence import representative_indices, diagnose_prompt_scale
            indices = representative_indices(test_data, args.equivalence_per_group)
            subset = torch.utils.data.Subset(test_data, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
            output_path = os.path.join(save_root, 'diagnostics', f'{args.testing_data}_prompt_scale.json')
            logger.info(f'Diagnosing {len(indices)} images; gates 1 vs 0.5; no training, HSF or full evaluation.')
            diagnose_prompt_scale(model, loader, logger, output_path)
            return
        if args.check_gate_equivalence:
            from tools.gate_equivalence import representative_indices, check_gate_equivalence
            indices = representative_indices(test_data, args.equivalence_per_group)
            logger.info(f'Checking {len(indices)} images; HSF retained with paired RNG; no training or full evaluation.')
            subset = torch.utils.data.Subset(test_data, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
            check_gate_equivalence(model, loader, logger)
            return
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(csv_root, exist_ok=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        save_fig_flag = save_fig

        metric_dict = model.evaluation(
            test_dataloader,
            test_data_cls_names,
            save_fig_flag,
            image_dir,
            gate_override=args.gate_override,
        )

        for tag, data in metric_dict.items():
            logger.info(
                '{:>15} \t\tI-Auroc:{:.2f} \tI-F1:{:.2f} \tI-AP:{:.2f} \tP-Auroc:{:.2f} \tP-F1:{:.2f} \tP-AP:{:.2f}'.
                    format(tag,
                           data['auroc_im'],
                           data['f1_im'],
                           data['ap_im'],
                           data['auroc_px'],
                           data['f1_px'],
                           data['ap_px'])
            )


        for k in metric_dict.keys():
            write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    elif args.testing_model == 'image':
        assert os.path.isfile(args.image_path), f"Please verify the input image path: {args.image_path}"
        ori_image = cv2.resize(cv2.imread(args.image_path), (args.image_size, args.image_size))
        pil_img = Image.open(args.image_path).convert('RGB')

        img_input = model.preprocess(pil_img).unsqueeze(0)
        img_input = img_input.to(model.device)

        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(
                img_input, [args.class_name], aggregation=True,
                gate_override=args.gate_override,
            )

        anomaly_map = anomaly_map[0, :, :]
        anomaly_score = anomaly_score[0]
        anomaly_map = anomaly_map.cpu().numpy()
        anomaly_score = anomaly_score.cpu().numpy()

        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = anomaly_map * 255
        anomaly_map = anomaly_map.astype(np.uint8)

        heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        vis_map = cv2.addWeighted(heat_map, 0.5, ori_image, 0.5, 0)

        vis_map = cv2.hconcat([ori_image, vis_map])
        save_path = os.path.join(args.save_path, args.save_name)
        print(f"Anomaly detection results are saved in {save_path}, with an anomaly of {anomaly_score:.3f} ")
        cv2.imwrite(save_path, vis_map)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP", add_help=True)

    # Paths and configurations
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_mvtec_colondb.pth',
                        help="Path to the pre-trained model (default: weights/pretrained_mvtec_colondb.pth)")

    parser.add_argument("--testing_model", type=str, default="dataset", choices=["dataset", "image"],
                        help="Model for testing (default: 'dataset')")

    # for the dataset model
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")

    # for the image model
    parser.add_argument("--image_path", type=str, default="asset/img.png",
                        help="Model for testing (default: 'asset/img.png')")
    parser.add_argument("--class_name", type=str, default="candle",
                        help="The class name of the testing image (default: 'candle')")
    parser.add_argument("--save_name", type=str, default="test.png",
                        help="Model for testing (default: 'dataset')")


    parser.add_argument("--save_path", type=str, default='./workspaces',
                        help="Directory to save results (default: './workspaces')")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")

    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations (default: False)")

    # Hyper-parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")

    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")
    parser.add_argument(
        "--fusion_mode", type=str, default="add",
        choices=["add", "layer_gate", "shared_gate"],
        help="Prompt fusion mode used by the checkpoint",
    )
    parser.add_argument(
        "--load_baseline", action="store_true",
        help="Explicitly import an add checkpoint into shared_gate (requires a fixed gate for evaluation)",
    )
    parser.add_argument(
        "--check_gate_equivalence", action="store_true",
        help="Compare add and shared_gate=1 outputs on representative images, then exit without training",
    )
    parser.add_argument(
        "--diagnose_source_losses", action="store_true",
        help="Diagnose classification/segmentation loss tradeoffs on source data only; no training",
    )
    parser.add_argument(
        "--compare_loss_shapes", action="store_true",
        help="Also compute batch-preserving loss from identical predictions; source diagnostic only",
    )
    parser.add_argument("--source_reference_report", default=None,
                        help="Previous source_gate_losses.json to verify identical configuration and sampled images")
    parser.add_argument(
        "--source_data", nargs='+', choices=['mvtec', 'colondb'], default=['mvtec', 'colondb'],
        help="Training-source datasets for loss diagnosis (never --testing_data)",
    )
    parser.add_argument("--source_samples_per_group", type=int, default=2,
                        help="Sample at most N images per source/class/label group")
    parser.add_argument("--source_sample_seed", type=int, default=111,
                        help="Local stratified sampling seed; inference keeps the existing test seed")
    parser.add_argument(
        "--diagnose_prompt_scale", action="store_true",
        help="Measure actual injected prompts before/after ln_1 at gates 1 and 0.5, without training",
    )
    parser.add_argument(
        "--equivalence_per_group", type=int, default=1,
        help="Images per (class, normal/anomalous) group for the equivalence check",
    )
    parser.add_argument(
        "--gate_override", type=float, default=None,
        help="For shared_gate, use a fixed weight in [0, 1] instead of the learned gate",
    )

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    train(args)
