"""RD: RT-DETR-l with the Ultralytics 8.4.7 trainer and validator (protocol amendment v1.2, section 11.3).

One correction, made in subclasses so that Ultralytics itself stays unedited. In 8.4.7,
`RTDETRValidator.postprocess` sorts the 300 query predictions of an image by score and then applies the
confidence mask computed in the unsorted query order:

    pred = pred[score.argsort(descending=True)]
    outputs[i] = pred[score > self.args.conf]

When a query scoring at or below conf (0.001) comes before a higher-scoring one, the wrong rows are kept: the count is
right, but a good detection can be dropped and a sub-threshold one kept instead (for example, scores [0.0005, 0.9,
0.5] keep 0.5 and 0.0005 and drop 0.9). COCO-trained RT-DETR-l keeps every query above 0.001, so the defect is
invisible there; a 9-class model trained on sparse radiographs can push background queries below it.
`RDValidator` applies the mask in the sorted order, which is what the code intends (later Ultralytics releases
rewrote this method). Everything else is the stock validator: square 640 input stretched without letterbox, no NMS,
one class per query, Ultralytics matching and metrics, Ultralytics fitness for the best epoch.

`RDTrainer` is the stock RT-DETR trainer with this validator (every epoch and the final evaluation of best.pt);
evaluation scripts pass `validator=RDValidator` to `RTDETR(...).val`. `vgra.infer.predict_rtdetr` (I1) makes the same
detections.
"""

from __future__ import annotations

from copy import copy

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import ops


class RDValidator(RTDETRValidator):
    """Ultralytics 8.4.7 RT-DETR validator with the confidence mask applied in the sorted order."""

    def postprocess(self, preds):
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference, tensor for exported models
            preds = [preds, None]
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            score, cls = scores[i].max(-1)  # (300,)
            pred = torch.cat([ops.xywh2xyxy(bbox), score[..., None], cls[..., None]], dim=-1)
            order = score.argsort(descending=True)  # sorted by confidence, as the metrics expect
            pred, score = pred[order], score[order]
            outputs.append(pred[score > self.args.conf])
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]


class RDTrainer(RTDETRTrainer):
    """Ultralytics 8.4.7 RT-DETR trainer that validates with `RDValidator`."""

    def get_validator(self):
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RDValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
