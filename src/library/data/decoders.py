from ..utils import *
import torch
from torch import Tensor
from collections import defaultdict


class Decoder:

    def __init__(self, args):
        self.label_map = args._r_labels
        self.part_map = args._r_parts
        self.anchor_name = args.anchor_name

        self.args = args
        self.down_ratio = args.down_ratio
        self.max_objects = args.max_objects  # K
        self.max_parts = args.max_parts  # P

    def decode_heatmaps(self, heatmaps: Tensor, offsets: Tensor, max_objects: int, embeddings: Tensor = None):
        heatmaps = clamped_sigmoid(heatmaps)  # (B, C, H/R, W/R)
        heatmaps = nms(heatmaps)  # (B, C, H/R, W/R)
        scores, inds, labels, ys, xs = topk(heatmaps, k=max_objects)  # (B, K)
        offsets = transpose_and_gather(offsets, inds)  # (B, K, 2)
        locs = torch.stack((xs, ys), dim=2) + offsets  # (B, K, 2)

        if embeddings is not None:
            offsets = transpose_and_gather(embeddings, inds)
            orgs = locs + offsets
            return locs, scores, labels, orgs

        return locs, scores, labels

    def associate(self, locs: Tensor, orgs: Tensor, scores: Tensor, p_scores: Tensor, dt: float):
        ct: float = self.args.conf_threshold

        locs = locs.unsqueeze(2).expand(-1, -1, self.max_parts, -1)  # (B, K, P, 2)
        orgs = orgs.unsqueeze(1).expand(-1, self.max_objects, -1, -1)  # (B, K, P, 2)
        sq_distance = torch.hypot(*torch.unbind(orgs - locs, dim=-1))  # (B, K, P)
        vals, inds = sq_distance.min(dim=1)  # (B, P)

        maps = []
        for b in range(input.size(0)):
            map = defaultdict(list)
            for p in range(self.max_parts):
                a = inds[b, p].item()
                if scores[b, a].item() >= ct and p_scores[b, p].item() >= ct and vals[b, p].item() <= dt:
                    map[a].append(p)
            maps.append(map)

        return maps

    def decode(self, input: Tensor):
        out_h, out_w = input["anchor_hm"].shape[2:]  # H/R, W/R
        in_h, in_w = int(self.down_ratio * out_h), int(self.down_ratio * out_w)  # H, W
        dt: float = self.args.decoder_dist_thresh * min(out_w, out_h)

        locs, scores, labels = self.decode_heatmaps(input["anchor_hm"], input["offsets"], self.max_objects)
        plocs, p_scores, p_labels, orgs = self.decode_heatmaps(input["part_hm"], input["offsets"], self.max_parts)
        maps = self.associate(locs, orgs, scores, p_scores, dt)  # (B, P)

        return [ImageAnnotation(
            image_path=f"batch_{b}",
            objects=[Object(
                name=self.label_map[labels[b, s].item()], 
                anchor=Keypoint(
                    kind=self.anchor_name,
                    x=locs[b, s, 0].item(), y=locs[b, s, 1].item(),
                    score=scores[b, s].item()), 
                parts=[Keypoint(
                    kind=self.part_map[p_labels[b, p].item()],
                    x=plocs[b, p, 0].item(), y=plocs[b, p, 1].item(),
                    score=p_scores[b, p].item()) for p in ps]
                ) for s, ps in map.items()],
            img_size=None).resize((out_w, out_h), (in_w, in_h)) for b, map in enumerate(maps)]

    def __call__(self, input: Tensor):
        return self.decode(input)


class KeypointDecoder:

    def __init__(self, args):
        self.label_map = args._r_labels
        self.part_map = args._r_parts
        self.anchor_name = args.anchor_name

        self.args = args
        self.down_ratio = args.down_ratio
        self.max_objects = args.max_objects  # K
        self.max_parts = args.max_parts  # P

    # output: (B, M+N+4, H/R, W/R), see network.py
    def __call__(self, outputs):
        conf_thresh = self.args.conf_threshold
        out_h, out_w = outputs["anchor_hm"].shape[2:]  # H/R, W/R
        in_h, in_w = int(self.down_ratio * out_h), int(self.down_ratio * out_w)  # H, W
        r_h, r_w = in_h / out_h, in_w / out_w

        # Anchors
        anchor_hm_sig = clamped_sigmoid(outputs["anchor_hm"])  # (B, M, H/R, W/R)
        anchor_hm = nms(anchor_hm_sig)  # (B, M, H/R, W/R)
        (anchor_scores, anchor_inds, anchor_labels, anchor_ys, anchor_xs) = topk(
            anchor_hm, k=self.max_objects)  # (B, K)
        anchor_offsets = transpose_and_gather(outputs["offsets"], anchor_inds)  # (B, K, 2)
        anchor_xs += anchor_offsets[..., 0]  # (B, K)
        anchor_ys += anchor_offsets[..., 1]  # (B, K)

        anchors = torch.stack((
            anchor_xs * r_w, anchor_ys * r_h,
            anchor_scores, anchor_labels.float()
        ), dim=2)  # (B, K, 4)

        # Parts
        part_hm_sig = clamped_sigmoid(outputs["part_hm"])  # (B, N, H/R, W/R)
        part_hm = nms(part_hm_sig)  # (B, N, H/R, W/R)
        (part_scores, part_inds, part_labels, part_ys, part_xs) = topk(
            part_hm, k=self.max_parts)  # (B, P)
        part_offsets = transpose_and_gather(outputs["offsets"], part_inds)  # (B, P, 2)
        part_xs += part_offsets[..., 0]  # (B, P)
        part_ys += part_offsets[..., 1]  # (B, P)

        parts = torch.stack((
            part_xs * r_w, part_ys * r_h,
            part_scores, part_labels.float(),
        ), dim=2)  # (B, P, 4)
        
        annotations = []

        for anchor_batch, part_batch in zip(anchors, parts):
            keypoints = []

            for x, y, score, label in anchor_batch:
                if score < conf_thresh: continue
                label = self.label_map[int(label.item())]
                keypoints.append(Keypoint(kind=label, x=x.item(), y=y.item(), score=score.item()))

            for x, y, score, label in part_batch:
                if score < conf_thresh: continue
                label = self.part_map[int(label)]
                keypoints.append(Keypoint(kind=label, x=x.item(), y=y.item(), score=score.item()))
            
            annotations.append(keypoints)

        return annotations