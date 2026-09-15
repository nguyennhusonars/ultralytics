# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 Detection Loss.

Adapted from the DEIMv2 repository.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.deimv2 import (
    bbox2distance,
    box_cxcywh_to_xyxy,
    weighting_function,
)
from ultralytics.utils.metrics import bbox_iou

from ultralytics.models.utils.ops import HungarianMatcher


def _box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in xyxy format."""
    from torchvision.ops.boxes import box_area

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def _generalized_box_iou(boxes1, boxes2):
    """Compute GIoU between two sets of boxes in xyxy format."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = _box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class DEIMv2DetectionLoss(nn.Module):
    """Loss function for DEIMv2 detection model.

    Computes MAL (Modified Adaptive Loss), bbox L1, GIoU, FGL (Fine-Grained Localization),
    and DDF (Decoupled Distillation Focal) losses.
    """

    def __init__(self, nc=80, alpha=0.75, gamma=1.5, reg_max=32, use_uni_set=True):
        super().__init__()
        self.nc = nc
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.use_uni_set = use_uni_set
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})

        # Weight dict for losses
        self.weight_dict = {
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        }

        # Caches
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None
        self.device = None

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """MAL (Modified Adaptive Loss) for classification."""
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = _box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.nc, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.nc + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute bbox L1 and GIoU losses."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(_generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
        ))
        return {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute FGL and DDF losses."""
        losses = {}
        if "pred_corners" not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
        ref_points = outputs["ref_points"][idx].detach()

        with torch.no_grad():
            is_dn = "is_dn" in outputs
            if is_dn and self.fgl_targets_dn is None:
                self.fgl_targets_dn = bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs["reg_scale"], outputs["up"],
                )
            if not is_dn and self.fgl_targets is None:
                self.fgl_targets = bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs["reg_scale"], outputs["up"],
                )

        target_corners, weight_right, weight_left = self.fgl_targets_dn if is_dn else self.fgl_targets

        ious = torch.diag(_box_iou(
            box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
            box_cxcywh_to_xyxy(target_boxes),
        )[0])
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses["loss_fgl"] = self._unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes,
        )

        if "teacher_corners" in outputs:
            pred_corners_all = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
            target_corners_all = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
            if not torch.equal(pred_corners_all, target_corners_all):
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
                weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                kl_loss = (
                    T ** 2
                    * nn.KLDivLoss(reduction="none")(
                        F.log_softmax(pred_corners_all / T, dim=1),
                        F.softmax(target_corners_all.detach() / T, dim=1),
                    ).sum(-1)
                )
                loss_match_local = weight_targets_local * kl_loss

                if "is_dn" not in outputs:
                    batch_scale = 8 / outputs["pred_boxes"].shape[0]
                    self.num_pos = (mask.sum() * batch_scale) ** 0.5
                    self.num_neg = ((~mask).sum() * batch_scale) ** 0.5
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_ddf"] = (
                    (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg)
                    / (self.num_pos + self.num_neg)
                )

        return losses

    @staticmethod
    def _unimodal_distribution_focal_loss(pred, label, weight_right, weight_left, weight=None, avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = (
            F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)
            + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        )
        if weight is not None:
            loss = loss * weight.float()
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss

    def _clear_cache(self):
        self.fgl_targets = None
        self.fgl_targets_dn = None
        self.num_pos = None
        self.num_neg = None

    def _get_go_indices(self, indices, indices_aux_list):
        """Get matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for row_col in unique_sorted:
                row_idx, col_idx = row_col[0].item(), row_col[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    @staticmethod
    def _get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device),
                ))
        return dn_match_indices

    def _match(self, outputs, targets):
        """Run Hungarian matching and convert to per-image indexing format."""
        pred_boxes = outputs["pred_boxes"].detach().contiguous()
        pred_logits = outputs["pred_logits"].detach().contiguous()

        gt_cls = torch.cat([t["labels"] for t in targets])
        gt_bboxes = torch.cat([t["boxes"] for t in targets])
        gt_groups = [len(t["labels"]) for t in targets]

        # Ultralytics matcher returns (query_idx, global_gt_idx) with cumulative offset
        raw_indices = self.matcher(pred_boxes, pred_logits, gt_bboxes, gt_cls, gt_groups)

        # Convert global gt indices to per-image indices
        cum_offset = 0
        indices = []
        for i, (query_idx, gt_idx) in enumerate(raw_indices):
            indices.append((query_idx, gt_idx - cum_offset))
            cum_offset += gt_groups[i]
        return indices

    def _compute_losses(self, outputs, targets, indices, num_boxes, losses_list=("mal", "boxes", "local")):
        """Compute losses for a single set of predictions."""
        losses = {}
        for loss_name in losses_list:
            if loss_name == "mal":
                l = self.loss_labels_mal(outputs, targets, indices, num_boxes)
            elif loss_name == "boxes":
                l = self.loss_boxes(outputs, targets, indices, num_boxes)
            elif loss_name == "local":
                l = self.loss_local(outputs, targets, indices, num_boxes)
            else:
                continue
            # Apply weight
            for k, v in l.items():
                if k in self.weight_dict:
                    losses[k] = v * self.weight_dict[k]
                else:
                    losses[k] = v
        return losses

    def forward(self, preds, batch):
        """Compute loss for DEIMv2.

        Args:
            preds: Dictionary of predictions from DEIMv2Decoder.
            batch: Dictionary containing ground truth data.

        Returns:
            (torch.Tensor): Total loss.
            (torch.Tensor): Main losses as a tensor [giou, cls, bbox] for logging.
        """
        self.device = preds["pred_boxes"].device
        self._clear_cache()

        # Build targets
        bs = batch["img"].shape[0]
        batch_idx = batch["batch_idx"]
        targets = []
        for i in range(bs):
            mask = batch_idx == i
            targets.append({
                "labels": batch["cls"][mask].to(torch.long).view(-1),
                "boxes": batch["bboxes"][mask],
            })

        # Main loss matching
        indices = self._match(preds, targets)

        # Get matching union set
        indices_aux_list = []
        cached_indices = []
        if "aux_outputs" in preds:
            aux_list = list(preds["aux_outputs"])
            if "pre_outputs" in preds:
                aux_list = aux_list + [preds["pre_outputs"]]
            for aux_outputs in aux_list:
                idx_aux = self._match(aux_outputs, targets)
                cached_indices.append(idx_aux)
                indices_aux_list.append(idx_aux)

        cached_indices_enc = []
        if "enc_aux_outputs" in preds:
            for enc_aux in preds["enc_aux_outputs"]:
                idx_enc = self._match(enc_aux, targets)
                cached_indices_enc.append(idx_enc)
                indices_aux_list.append(idx_enc)

        if indices_aux_list:
            indices_go = self._get_go_indices(indices, indices_aux_list)
            num_boxes_go = max(sum(len(x[0]) for x in indices_go), 1)
        else:
            indices_go = indices
            num_boxes_go = max(sum(len(t["labels"]) for t in targets), 1)

        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # Main losses
        losses = {}

        # mal loss uses per-layer indices, boxes/local use indices_go (union set)
        mal_losses = self.loss_labels_mal(preds, targets, indices, num_boxes)
        for k, v in mal_losses.items():
            if k in self.weight_dict:
                losses[k] = v * self.weight_dict[k]

        box_idx = indices_go if self.use_uni_set else indices
        box_num = num_boxes_go if self.use_uni_set else num_boxes
        box_losses = self.loss_boxes(preds, targets, box_idx, box_num)
        for k, v in box_losses.items():
            if k in self.weight_dict:
                losses[k] = v * self.weight_dict[k]

        local_losses = self.loss_local(preds, targets, box_idx, box_num)
        for k, v in local_losses.items():
            if k in self.weight_dict:
                losses[k] = v * self.weight_dict[k]

        # Auxiliary losses
        if "aux_outputs" in preds:
            for i, aux_outputs in enumerate(preds["aux_outputs"]):
                if "pred_corners" in aux_outputs:
                    aux_outputs["up"] = preds["up"]
                    aux_outputs["reg_scale"] = preds["reg_scale"]
                # MAL uses per-layer cached indices; boxes/local use union set
                mal_idx = cached_indices[i] if i < len(cached_indices) else indices
                aux_mal = self.loss_labels_mal(aux_outputs, targets, mal_idx, num_boxes)
                for k, v in aux_mal.items():
                    w = self.weight_dict.get(k, 1.0)
                    losses[f"{k}_aux_{i}"] = v * w

                if self.use_uni_set:
                    aux_box = self.loss_boxes(aux_outputs, targets, indices_go, num_boxes_go)
                    aux_local = self.loss_local(aux_outputs, targets, indices_go, num_boxes_go)
                else:
                    aux_box = self.loss_boxes(aux_outputs, targets, mal_idx, num_boxes)
                    aux_local = self.loss_local(aux_outputs, targets, mal_idx, num_boxes)
                for k, v in {**aux_box, **aux_local}.items():
                    w = self.weight_dict.get(k, 1.0)
                    losses[f"{k}_aux_{i}"] = v * w

        # Pre outputs
        if "pre_outputs" in preds:
            pre_idx = cached_indices[-1] if cached_indices else indices
            pre_l = self._compute_losses(preds["pre_outputs"], targets, pre_idx, num_boxes,
                                          losses_list=("mal", "boxes"))
            for k, v in pre_l.items():
                losses[f"{k}_pre"] = v

        # Encoder aux losses
        if "enc_aux_outputs" in preds:
            class_agnostic = preds.get("enc_meta", {}).get("class_agnostic", False)
            if class_agnostic:
                orig_nc = self.nc
                self.nc = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, enc_aux in enumerate(preds["enc_aux_outputs"]):
                enc_idx = cached_indices_enc[i] if i < len(cached_indices_enc) else indices
                # MAL uses per-layer indices
                enc_mal = self.loss_labels_mal(enc_aux, enc_targets, enc_idx, num_boxes)
                for k, v in enc_mal.items():
                    w = self.weight_dict.get(k, 1.0)
                    losses[f"{k}_enc_{i}"] = v * w
                # Boxes use union set if enabled
                if self.use_uni_set:
                    enc_box = self.loss_boxes(enc_aux, enc_targets, indices_go, num_boxes_go)
                else:
                    enc_box = self.loss_boxes(enc_aux, enc_targets, enc_idx, num_boxes)
                for k, v in enc_box.items():
                    w = self.weight_dict.get(k, 1.0)
                    losses[f"{k}_enc_{i}"] = v * w

            if class_agnostic:
                self.nc = orig_nc

        # DN losses
        if "dn_outputs" in preds and "dn_meta" in preds:
            dn_meta = preds["dn_meta"]
            indices_dn = self._get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta["dn_num_group"]

            for i, dn_aux in enumerate(preds["dn_outputs"]):
                dn_aux["is_dn"] = True
                dn_aux["up"] = preds["up"]
                dn_aux["reg_scale"] = preds["reg_scale"]
                dn_l = self._compute_losses(dn_aux, targets, indices_dn, dn_num_boxes)
                for k, v in dn_l.items():
                    losses[f"{k}_dn_{i}"] = v

            if "dn_pre_outputs" in preds:
                dn_pre_l = self._compute_losses(
                    preds["dn_pre_outputs"], targets, indices_dn, dn_num_boxes,
                    losses_list=("mal", "boxes"),
                )
                for k, v in dn_pre_l.items():
                    losses[f"{k}_dn_pre"] = v

        # Clean up NaN
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

        total_loss = sum(losses.values())
        # Return main 3 losses for logging: [giou_loss, cls_loss, bbox_loss]
        main_losses = torch.as_tensor(
            [
                losses.get("loss_giou", torch.tensor(0.0, device=self.device)).detach(),
                losses.get("loss_mal", torch.tensor(0.0, device=self.device)).detach(),
                losses.get("loss_bbox", torch.tensor(0.0, device=self.device)).detach(),
            ],
            device=self.device,
        )
        return total_loss, main_losses
