import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from loss import FocalLoss, BinaryDiceLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .adaclip import *
from .custom_clip import create_model_and_transforms


class AdaCLIP_Trainer(nn.Module):
    def __init__(
            self,
            # clip-related
            backbone, feat_list, input_dim, output_dim,

            # learning-related
            learning_rate, device, image_size,

            # model settings
            prompting_depth=3, prompting_length=2,
            prompting_branch='VL', prompting_type='SD',
            use_hsf=True, k_clusters=20,
            fusion_mode="add", gate_learning_rate=0.001,
    ):

        super(AdaCLIP_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.fusion_mode = fusion_mode
        if gate_learning_rate <= 0:
            raise ValueError("gate_learning_rate must be positive.")
        self.gate_learning_rate = gate_learning_rate
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        ########### different model choices
        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip  = freeze_clip.to(device)
        freeze_clip.eval()

        self.clip_model = AdaCLIP(freeze_clip=freeze_clip,
                                  text_channel=output_dim,
                                  visual_channel=input_dim,
                                  prompting_length=prompting_length,
                                  prompting_depth=prompting_depth,
                                  prompting_branch=prompting_branch,
                                  prompting_type=prompting_type,
                                  use_hsf=use_hsf,
                                  k_clusters=k_clusters,
                                  output_layers=feat_list,
                                  device=device,
                                  image_size=image_size,
                                  fusion_mode=self.fusion_mode).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        # update parameters
        self.base_parameter_names = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'dynamic_visual_prompt_generator',
            'dynamic_text_prompt_generator',
        ]

        self.gate_parameter_names = [
            "visual_gate_generator",
            "text_gate_generator",
            "shared_gate_generator",
        ]

        self.learnable_paramter_list = (
            self.base_parameter_names
            + self.gate_parameter_names
        )

        base_parameters = []
        gate_parameters = []

        for name, parameter in self.clip_model.named_parameters():
            if any(gate_name in name for gate_name in self.gate_parameter_names):
                gate_parameters.append(parameter)
            elif any(base_name in name for base_name in self.base_parameter_names):
                base_parameters.append(parameter)

        self.params_to_update = base_parameters + gate_parameters

        optimizer_groups = [
            {
                "params": base_parameters,
                "lr": learning_rate,
                "weight_decay": 0.01,
            },
        ]

        if gate_parameters:
            optimizer_groups.append(
                {
                    "params": gate_parameters,
                    "lr": self.gate_learning_rate,
                    "weight_decay": 0.0,
                }
            )

        # build the optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.5, 0.999),
        )

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    # print(f'{param}: {update_name}')
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        has_shared_gate = any(
            "shared_gate_generator" in name for name in state_dict
        )
        if has_shared_gate and self.fusion_mode != "shared_gate":
            raise ValueError(
                "This checkpoint uses shared_gate; pass --fusion_mode shared_gate"
            )
        incompatible = self.load_state_dict(state_dict, strict=False)
        if self.fusion_mode == "shared_gate":
            missing_gate = [
                name for name in incompatible.missing_keys
                if "shared_gate_generator" in name
            ]
            if missing_gate:
                raise ValueError(
                    f"Checkpoint {path} is missing shared gate weights: {missing_gate}"
                )

    def detection_loss(self, anomaly_map, anomaly_score, items):
        if not isinstance(anomaly_map, list):
            anomaly_map = [anomaly_map]

        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze()

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items['anomaly'].to(self.device).float()
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0
        loss = 0

        # classification loss
        classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
        loss += classification_loss

        # seg loss
        seg_loss = 0
        for am, in zip(anomaly_map):
            seg_loss += (self.loss_focal(am, gt) + self.loss_dice(am[:, 1, :, :], gt) +
                         self.loss_dice(am[:, 0, :, :], 1-gt))

        loss += seg_loss
        return loss

    def train_one_batch(self, items, gate_override=None):
        image = items['img'].to(self.device)
        cls_name = items['cls_name']
        anomaly_map, anomaly_score = self.clip_model(
            image, cls_name, aggregation=False, gate_override=gate_override
        )
        loss = self.detection_loss(anomaly_map, anomaly_score, items)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def prepare_shared_gate_training(self):
        if self.fusion_mode != "shared_gate":
            raise ValueError("Gate utility training requires fusion_mode='shared_gate'")
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.clip_model.shared_gate_generator.parameters():
            parameter.requires_grad_(True)

    def train_shared_gate_batch(self, items, utility_temperature, task_weight):
        image = items['img'].to(self.device)
        if image.shape[0] != 1:
            raise ValueError("Shared gate utility training currently requires batch size 1")
        cls_name = items['cls_name']

        with torch.no_grad():
            static_map, static_score = self.clip_model(
                image, cls_name, aggregation=False, gate_override=0.0
            )
            static_loss = self.detection_loss(static_map, static_score, items)
            dynamic_map, dynamic_score = self.clip_model(
                image, cls_name, aggregation=False, gate_override=1.0
            )
            dynamic_loss = self.detection_loss(dynamic_map, dynamic_score, items)
            utility_target = torch.sigmoid(
                (static_loss - dynamic_loss) / utility_temperature
            )

        gated_map = gated_score = None
        if task_weight > 0:
            gated_map, gated_score = self.clip_model(
                image, cls_name, aggregation=False
            )
            gate_logits = self.clip_model.shared_gate_logits
        else:
            gate_logits, _ = self.clip_model.shared_gate_generator(
                self.clip_model.condition_features
            )
        utility_loss = F.binary_cross_entropy_with_logits(
            gate_logits.reshape(-1), utility_target.reshape(-1)
        )
        loss = utility_loss
        if task_weight > 0:
            loss = loss + task_weight * self.detection_loss(
                gated_map, gated_score, items
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, loader, stage="standard", utility_temperature=1.0,
                    task_weight=1.0):
        if stage == "gate":
            self.clip_model.eval()
        else:
            self.clip_model.train()
        loss_list = []
        for items in loader:
            if stage == "dual":
                gate_override = float(torch.randint(0, 2, ()).item())
                loss = self.train_one_batch(items, gate_override=gate_override)
            elif stage == "gate":
                loss = self.train_shared_gate_batch(
                    items, utility_temperature, task_weight
                )
            elif stage == "standard":
                loss = self.train_one_batch(items)
            else:
                raise ValueError(f"Unknown training stage: {stage}")
            loss_list.append(loss.item())

        return np.mean(loss_list)

    @staticmethod
    def summarize_gate_values(gate_values):
        gate_statistics = {}

        for branch, collected_gates in gate_values.items():
            if not collected_gates:
                continue

            gates = torch.cat(collected_gates, dim=0)
            branch_statistics = []

            for layer in range(gates.shape[1]):
                values = gates[:, layer]
                branch_statistics.append(
                    {
                        'mean': values.mean().item(),
                        'std': values.std(unbiased=False).item(),
                        'min': values.min().item(),
                        'max': values.max().item(),
                        'p01': torch.quantile(values, 0.01).item(),
                        'p99': torch.quantile(values, 0.99).item(),
                        'near0': (values < 0.05).float().mean().item(),
                        'near2': (values > 1.95).float().mean().item(),
                    }
                )

            gate_statistics[branch] = branch_statistics

        return gate_statistics

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None,
                   collect_gate_stats=False, gate_override=None):
        self.clip_model.eval()

        self.last_gate_statistics = {}
        gate_values = None
        if collect_gate_stats:
            if self.fusion_mode == "shared_gate":
                gate_values = {'shared': []}
            else:
                gate_values = {'visual': [], 'text': []}

        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_indx = 0
            for indx, items in enumerate(dataloader):
                if save_fig:
                    path = items['img_path']
                    for _path in path:
                        vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                        results['imgs'].append(vis_image)
                    cls_name = items['cls_name']
                    for _cls_name in cls_name:
                        image_indx += 1
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                results['cls_names'].extend(cls_name)
                gt_mask = items['img_mask']
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                for _gt_mask in gt_mask:
                    results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())  # px

                # pixel level
                anomaly_map, anomaly_score = self.clip_model(
                    image, cls_name, aggregation=True,
                    gate_override=gate_override,
                )

                if gate_values is not None:
                    if self.fusion_mode == "shared_gate":
                        gate_values['shared'].append(
                            self.clip_model.shared_gate_weights.detach().float().cpu()
                        )
                    else:
                        visual_gates = self.clip_model.visual_prompter.dynamic_gates
                        text_gates = self.clip_model.text_prompter.dynamic_gates

                        if visual_gates is not None:
                            gate_values['visual'].append(
                                visual_gates.detach().float().cpu()
                            )

                        if text_gates is not None:
                            gate_values['text'].append(
                                text_gates.detach().float().cpu()
                            )

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results['anomaly_maps'].append(_anomaly_map)
                    results['anomaly_scores'].append(_anomaly_score)

                is_anomaly = np.array(items['anomaly'])
                for _is_anomaly in is_anomaly:
                    results['imgs_gts'].append(_is_anomaly)

        # visualization
        if save_fig:
            print('saving fig.....')
            visualization.plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'AdaCLIP': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir
            )

        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()

        for obj in obj_list:
            metric = calculate_metric(results, obj)
            obj_full_name = f'{obj}'
            metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        if gate_values is not None:
            self.last_gate_statistics = self.summarize_gate_values(gate_values)

        return metric_dict
