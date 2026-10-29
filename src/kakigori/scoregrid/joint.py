"""Joint fine-tune of the detector with the scoregrid decoder: does the
transcription loss move the detector where the GNN's edge loss did not?

Per page, the GNN wrapper's detector (backbone, neck) runs with gradients:
its heads take the page's detection loss on the ground-truth boxes, and
every decoder system of the page (scoregrid data: the cached detected nodes
and the target grid) gets its RoI features and GNN states recomputed from
those same features, then the decoder's flow loss:

    L = det_weight * L_det + flow_weight * mean(L_flow)

The nodes stay the starting detector's cached detections (as in stage D);
what the decoder loss moves is the features they are read from. The GNN
joint (stage C) trained its edge loss on ground-truth boxes at a vision lr
of 1e-5: nothing tied the detector to its own mistakes, and mAP stayed put.

Arms, from the same checkpoints, steps and learning rates:
  --flow-weight 0          detection-only control (what the lite2 labels alone do)
  default                  joint
  --train-detector none    decoder-only control (detector, GNN frozen)
Detection mAP: eval-model on the output (a GNN wrapper checkpoint); the
transcription: build-scoregrid-data --checkpoint OUT --detector-checkpoint OUT,
then eval-scoregrid --checkpoint OUT/decoder.pt.
"""
# Standard library imports
import json
import math
import random
import logging
import argparse
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import save_file

# First party imports
from kakigori.graph.model import node_features
from kakigori.graph.heuristics import generate_axis_aware_edges
from kakigori.graph.validate_predictions import _load_gnn
from kakigori.vision.losses import CIoULoss, DynamicFocalLoss
from kakigori.vision.dataset import letterbox_image, page_to_canvas
from kakigori.vision.trainer import OMRTrainer, regression_loss

# Local folder imports
from .model import ScoreGridDecoder, flow_loss, grid_tensors
from .train import N_MAX, load_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def node_boxes(record: dict, geo: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """A record's node boxes and system box on the letterboxed canvas, from its
    stored geometry: centre and size normalized to the nodes' x span and the
    system's height (page_systems), both stored in page pixels."""
    x1, _, x2, _ = page_to_canvas(torch.tensor([[record["x_span"][0], 0.0, record["x_span"][1], 0.0]]), geo)[0]
    system = page_to_canvas(torch.tensor([record["system_box"]], dtype=torch.float32), geo)[0]
    sy1, sy2 = system[1], system[3]
    sw, sh = max(float(x2 - x1), 1e-3), max(float(sy2 - sy1), 1e-3)
    g = record["geom"].float()
    cx, cy = x1 + g[:, 0] * sw, sy1 + g[:, 1] * sh
    w, h = g[:, 2] * sw, g[:, 3] * sh
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1), system


def collate_live(records: list, device) -> dict:
    """scoregrid.train.collate for records whose roi / gnn are live (on the
    device, with gradients): the same padding and truncation."""
    n = min(max(len(r["cls"]) for r in records), N_MAX)
    b = len(records)
    z = lambda *shape, **kw: torch.zeros(*shape, device=device, **kw)
    out = {
        "cls": z(b, n, dtype=torch.long), "geom": z(b, n, 5),
        "row": torch.full((b, n), -1, dtype=torch.long, device=device),
        "meas": torch.full((b, n), -1, dtype=torch.long, device=device),
        "conf": z(b, n), "synth": z(b, n), "roi": z(b, n, 256), "gnn": z(b, n, 256),
        "cand": z(b, n, dtype=torch.long), "meter": z(b, 2, dtype=torch.long),
        "valid": z(b, n, dtype=torch.bool),
    }
    for i, r in enumerate(records):
        total = len(r["cls"])
        if total <= n:
            sel, k = torch.arange(total), total
        else:
            # synthesized structure nodes (bbox_repair, at the tail) + the earliest
            sel, k = torch.argsort(r["synth"].int(), descending=True, stable=True)[:n].sort().values, n
        for key, dtype in (("cls", torch.long), ("row", torch.long), ("meas", torch.long),
                           ("geom", torch.float32), ("conf", torch.float32), ("synth", torch.float32),
                           ("cand", torch.long)):
            if key in r:
                out[key][i, :k] = r[key][sel].to(device, dtype)
        out["roi"][i, :k] = r["roi"][sel.to(r["roi"].device)].float()
        out["gnn"][i, :k] = r["gnn"][sel.to(r["gnn"].device)].float()
        if "meter" in r:
            out["meter"][i] = torch.tensor(r["meter"], device=device)
        out["valid"][i, :k] = True
    cells, barline, staff = zip(*(grid_tensors(r["target"]) for r in records), strict=True)
    out["cells"] = torch.stack(cells).to(device)
    out["barline"] = torch.stack(barline).to(device)
    out["staff"] = torch.stack(staff).to(device)
    return out


def pages_of(pairs: list[str]) -> list[tuple]:
    """(dataset dir, score, page index, records) per page of the matched
    scoregrid records; ``pairs`` are SCOREGRID_DIR=DATASET_DIR."""
    pages = []
    for pair in pairs:
        sg_dir, data_dir = (Path(p) for p in pair.split("=", 1))
        by_page = defaultdict(list)
        for r in load_records(sg_dir):
            by_page[(r["score"], r["page"])].append(r)
        for (score, page), records in sorted(by_page.items()):
            pages.append((data_dir, score, page, sorted(records, key=lambda r: r["order"])))
        logger.info(f"{sg_dir}: {sum(len(p[3]) for p in pages)} systems on {len(pages)} pages so far")
    return pages


class Joint:
    def __init__(self, wrapper, decoder, class_list, args, device):
        self.wrapper, self.decoder, self.device, self.args = wrapper, decoder, device, args
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.focal = DynamicFocalLoss(base_gamma=args.base_gamma, max_gamma=args.max_gamma)
        self.ciou = CIoULoss()
        self.targets = SimpleNamespace(scale_ranges=wrapper.detector.scale_ranges)

    def detection_loss(self, fused, ann_path: Path, geo: dict, canvas, progress: float):
        """The page's detection loss on its ground-truth boxes (train_joint's)."""
        det = self.wrapper.detector
        anns = [a for a in json.loads(ann_path.read_text())["annotations"] if a["class"] in self.class_to_idx]
        boxes = page_to_canvas(torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4), geo)
        labels = torch.tensor([self.class_to_idx[a["class"]] for a in anns], dtype=torch.long)
        cw, ch = canvas
        cxcywh = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / cw, (boxes[:, 1] + boxes[:, 3]) / 2 / ch,
                              (boxes[:, 2] - boxes[:, 0]) / cw, (boxes[:, 3] - boxes[:, 1]) / ch], dim=1)
        outputs = []
        for feature, head in zip(fused, det.heads):
            cls_out, reg_out = head(feature)
            outputs.append({"cls": cls_out.float(), "reg": reg_out.float(), "reg_slot": det.reg_slot})
        num_classes = det.heads[0].cls_branch[-1].out_channels
        sizes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs]
        cls_t, reg_t, masks = OMRTrainer.build_targets(
            self.targets, [{"boxes": cxcywh.to(self.device), "labels": labels.to(self.device)}],
            sizes, num_classes, self.device, canvas=canvas, reg_slot=det.reg_slot)
        cls_loss = sum(self.focal(o["cls"], t, progress=progress) for o, t in zip(outputs, cls_t))
        reg_loss = regression_loss(outputs, reg_t, masks, self.ciou)
        return self.args.cls_weight * cls_loss + self.args.reg_weight * reg_loss

    def live_records(self, fused, records: list, geo: dict, canvas_hw) -> list:
        """The page's decoder records with RoI features and GNN states
        recomputed from ``fused`` (with gradients)."""
        wrapper = self.wrapper
        feat_dict = {str(i): f for i, f in enumerate(fused)}
        live = []
        for r in records:
            boxes, system = node_boxes(r, geo)
            boxes = boxes.to(self.device)
            labels = r["cls"].long().to(self.device)
            roi = wrapper.roi_extractor(feat_dict, [boxes], canvas_hw)
            rel = boxes.clone()
            rel[:, [0, 2]] -= float(system[0])
            rel[:, [1, 3]] -= float(system[1])
            x = node_features(wrapper.gnn, roi, rel / geo["canvas_w"], labels,
                              scores=r["conf"].float().to(self.device),
                              synthesized=r["synth"].float().to(self.device))
            cand = generate_axis_aware_edges(rel.detach(), labels, self.class_to_idx)
            h = wrapper.gnn.node_states(x, cand) if cand.shape[1] else torch.zeros_like(roi)
            live.append({**r, "roi": roi, "gnn": h})
        return live

    def page_loss(self, page: tuple, progress: float, flow_always: bool = False) -> tuple:
        data_dir, score, page_idx, records = page
        stem = f"{score}_page{page_idx + 1}"
        det = self.wrapper.detector
        canvas_img, geo = letterbox_image(Image.open(data_dir / "imgs" / f"{stem}.png").convert("L"),
                                          det.input_size)
        img = TF.to_tensor(canvas_img).to(self.device).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.set_grad_enabled(self.args.train_detector == "full"):
                features = det.backbone(img)
            fused = det.neck(features)
            det_loss = torch.zeros((), device=self.device)
            if self.args.det_weight > 0 and self.args.train_detector != "none":
                det_loss = self.detection_loss(fused, data_dir / "annotations" / f"{stem}.json", geo,
                                               det.input_size, progress)
            flow = torch.zeros((), device=self.device)
            if (self.args.flow_weight > 0 or flow_always) and records:
                batch = collate_live(self.live_records(fused, records, geo, img.shape[-2:]), self.device)
                flow, _ = flow_loss(self.decoder, batch["cells"], batch["barline"], batch["staff"], batch)
        return det_loss.float(), flow.float()


def trainable(args, wrapper, decoder) -> list:
    """Optimizer groups for the arm: what trains, at which learning rate."""
    det = wrapper.detector
    for p in wrapper.parameters():
        p.requires_grad = False
    groups = []
    if args.train_detector in ("neck", "full"):
        parts = [(det.neck, args.vision_lr), (det.heads, args.head_lr)]
        if args.train_detector == "full":
            parts.append((det.backbone, args.vision_lr))
        for module, lr in parts:
            params = list(module.parameters())
            for p in params:
                p.requires_grad = True
            groups.append({"params": params, "lr": lr})
    if args.train_gnn:
        params = list(wrapper.gnn.parameters())
        for p in params:
            p.requires_grad = True
        groups.append({"params": params, "lr": args.gnn_lr})
    for p in decoder.parameters():
        p.requires_grad = args.flow_weight > 0
    if args.flow_weight > 0:
        groups.append({"params": list(decoder.parameters()), "lr": args.decoder_lr})
    return groups


def save(out: Path, step: int, wrapper, decoder, decoder_config: dict, args) -> None:
    ckpt = out / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in wrapper.state_dict().items()},
              str(ckpt / "model.safetensors"))
    torch.save({"model": decoder.state_dict(), "config": decoder_config, "step": step}, out / "decoder.pt")
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Joint detector + scoregrid decoder fine-tune.")
    parser.add_argument("--checkpoint", required=True, help="GNN wrapper run (its detector trains)")
    parser.add_argument("--decoder", required=True, help="scoregrid decoder (best.pt)")
    parser.add_argument("--train", nargs="+", required=True, help="SCOREGRID_DIR=DATASET_DIR")
    parser.add_argument("--val", nargs="+", default=[], help="SCOREGRID_DIR=DATASET_DIR (flow loss)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="conf-lite/config.json")
    parser.add_argument("--train-detector", choices=("none", "neck", "full"), default="full")
    parser.add_argument("--train-gnn", action="store_true", help="the GNN trains too (else gradients only)")
    parser.add_argument("--det-weight", type=float, default=0.5)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=5.0)
    parser.add_argument("--base-gamma", type=float, default=2.0)
    parser.add_argument("--max-gamma", type=float, default=3.0)
    parser.add_argument("--vision-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--gnn-lr", type=float, default=3e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--pages-per-step", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--val-pages", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    class_list = json.load(open(args.config))["target_classes"]
    wrapper = _load_gnn(args.checkpoint, device)
    wrapper.eval()  # LayerNorm detector, no dropout: eval mode with gradients
    state = torch.load(args.decoder, map_location=device, weights_only=False)
    decoder = ScoreGridDecoder(**state["config"]).to(device)
    decoder.load_state_dict(state["model"])
    decoder.train()

    train_pages, val_pages = pages_of(args.train), pages_of(args.val)
    rng.shuffle(val_pages)
    val_pages = val_pages[:args.val_pages]
    groups = trainable(args, wrapper, decoder)
    if not groups:
        raise SystemExit("nothing to train for this arm")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in groups]
    params = [p for g in groups for p in g["params"]]
    joint = Joint(wrapper, decoder, class_list, args, device)
    logger.info(f"arm: train_detector={args.train_detector} train_gnn={args.train_gnn} "
                f"det_weight={args.det_weight} flow_weight={args.flow_weight}; "
                f"{sum(p.numel() for p in params) / 1e6:.1f}M parameters train; "
                f"{len(train_pages)} train pages, {len(val_pages)} val pages")

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))))

    def validate() -> float:
        decoder.eval()
        total = 0.0
        with torch.no_grad():
            for page in val_pages:
                # the decoder's loss on the current features, whatever the arm trains
                total += float(joint.page_loss(page, 1.0, flow_always=True)[1])
        decoder.train()
        return total / max(1, len(val_pages))

    log = open(out / "train.log", "a")
    order, cursor = list(range(len(train_pages))), 0
    rng.shuffle(order)
    msg = f"step 0 val flow {validate():.4f}"
    logger.info(msg)
    log.write(msg + "\n")
    sums = defaultdict(float)
    for step in range(1, args.steps + 1):
        for g, base in zip(optimizer.param_groups, base_lrs, strict=True):
            g["lr"] = base * lr_scale(step - 1)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.pages_per_step):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            det_loss, flow = joint.page_loss(train_pages[order[cursor]], step / args.steps)
            cursor += 1
            loss = (args.det_weight * det_loss + args.flow_weight * flow) / args.pages_per_step
            if loss.requires_grad:
                loss.backward()
            sums["det"] += float(det_loss) / args.pages_per_step
            sums["flow"] += float(flow) / args.pages_per_step
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % 50 == 0:
            msg = f"step {step}/{args.steps} det {sums['det'] / 50:.4f} flow {sums['flow'] / 50:.4f}"
            logger.info(msg)
            log.write(msg + "\n")
            log.flush()
            sums.clear()
        if step % args.eval_every == 0 or step == args.steps:
            msg = f"step {step} val flow {validate():.4f}"
            logger.info(msg)
            log.write(msg + "\n")
            log.flush()
            save(out, step, wrapper, decoder, state["config"], args)
    logger.info(f"done -> {out}")


if __name__ == "__main__":
    main()
