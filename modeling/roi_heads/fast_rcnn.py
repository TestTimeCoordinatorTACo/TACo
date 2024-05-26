# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F
import copy
from .utils.converter import coco2bb
from .coco_evaluator import get_coco_metrics
from .utils.enumerators import BBType
import numpy as np

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)

# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        #print("in focal layer,",proposals)# normal has gt classes
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes
        #print("in focal loss,",proposals)# normal has gt classes
        #print("in focal loss,",self.proposals)# no gt classes since they are in self.gt_classes 
    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()

class CalibrationLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        gt_im_id,
        pred_instances,
        targets,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(CalibrationLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes
        self.orig_proposals = proposals
        self.gt_im_id = gt_im_id
        self.gt_instances = targets
        self.pred_instances = pred_instances
    def losses(self):
        return {
            "cali_loss": self.comput_cali_loss(),
        }
    
    #def filter_pseudo_GT(self, targ_, results_):# it is actually getting the higher iou TP bbox (will the bbox usually have higher confidence?) (but what if very few high ious? -> pick the top 1 iou?) -> more likely to be real TP (but if the confidence are all high, there will be no accurate low confi -> good thing, no need to calibrate; if the iou is too high to ignore some TP, this will lead to wrong FP-> so need to lower the iou when calculating TP,FP stuff) so need to simulate (keep some very low iou positives, and append the resulting false positive number by randomness with low iou)
    #    # so filter through iou first then observe the scores of matched results to see whether there will be lack of nAI and nIC

        
    def comput_cali_loss(self):
        #print(self.orig_proposals[0])#normal bbox size (x1,y1,x2,y2)
        #print(self.gt_instances[0])#normal bbox size (x1,y1,x2,y2)
        #print(len(self.num_preds_per_image))#16
        #print(sum(self.num_preds_per_image))#8192
        #print(len(self.pred_class_logits))#8192
        #print(len(self.pred_proposal_deltas), self.pred_proposal_deltas[0].shape)#8192, 320 (4*80)# we need 8192, 4
        #print(self.pred_class_logits)#8192, 80# we need 8192, 1 -> how they choose 1 from 80 during inference?
        #exit()
        
        cat_l = [{'id': 24, 'name': 'person', 'supercategory': 'human'}, {'id': 25, 'name': 'rider', 'supercategory': 'human'}, {'id': 26, 'name': 'car', 'supercategory': 'vehicle'}, {'id': 27, 'name': 'truck', 'supercategory': 'vehicle'}, {'id': 28, 'name': 'bus', 'supercategory': 'vehicle'}, {'id': 31, 'name': 'train', 'supercategory': 'vehicle'}, {'id': 32, 'name': 'motorcycle', 'supercategory': 'vehicle'}, {'id': 33, 'name': 'bicycle', 'supercategory': 'vehicle'}]
        inverse_id_map={0: 24, 1: 25, 2: 26, 3: 27, 4: 28, 5: 31, 6: 32, 7: 33}
        preds_json = []
        box_id_pr = 0

        pr_data_json = {}
        pr_data_json["images"] = []
        pr_data_json["annotations"] = []
        pr_data_json["categories"] = cat_l

        gt_data_json = {}
        gt_data_json["images"] = []
        gt_data_json["annotations"] = []
        gt_data_json["categories"] = cat_l

        scores=[instancei.scores for instancei in self.pred_instances]
        labels=[instancei.pred_classes for instancei in self.pred_instances]
        boxes=[instancei.pred_boxes for instancei in self.pred_instances]
        results_ = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        #print(self.pred_instances[0], self.pred_instances[0].image_size)

        res = {target.item(): output for target, output in zip(self.gt_im_id, results_)}
        
        for d ,iml in enumerate(res.keys()):
            pr_data_json["images"].append({
                "file_name":str(iml).zfill(12) + ".jpg",
                "id":iml,
                "height":self.pred_instances[d].image_size[0],
                "width":self.pred_instances[d].image_size[1]
            })

        gt_data_json["images"] = pr_data_json["images"]
        
        for im_id, p in res.items():
            for s, l, b in zip(p['scores'], p['labels'], p['boxes']):
                #print(b)
                
                pr_data_json["annotations"].append({
                        "id": box_id_pr,
                        "segmentation": [],
                        "image_id": im_id,
                        "bbox": b,
                        "category_id": inverse_id_map[l.item()],
                        "score": s
                    })
                box_id_pr += 1
        #print(self.pred_instances[0])
        targ = copy.deepcopy(self.gt_instances)
        t_labels=[instancei.gt_classes for instancei in targ]
        t_boxes=[instancei.gt_boxes for instancei in targ]       
        targ_ = [{'labels': l, 'boxes': b} for l, b in zip(t_labels, t_boxes)]
        #filtered_targ = self.filter_pseudo_GT(targ_, results_)

        box_id_g = 0
        for b_id, p in enumerate(targ_):
            for l, b in zip(p['labels'], p['boxes']):
                gt_data_json["annotations"].append({
                    "id": box_id_g,
                    "segmentation": [],
                    "image_id": self.gt_im_id[b_id].item(),
                    "bbox": b,
                    "category_id": inverse_id_map[l.item()],
                    "area": (b[2]-b[0])*(b[3]-b[1])
                })
                #print(gt_data_json)
                
                box_id_g += 1
                
        pred_to = coco2bb(pr_data_json, BBType.DETECTED)
        gt_to = coco2bb(gt_data_json, BBType.GROUND_TRUTH)

        # Categorize all the dets into precision space
        fin_res = get_coco_metrics(
        gt_to,
        pred_to,
        iou_threshold=0.5,#0.5
        area_range=(0, np.inf),
        max_dets=100)
        #print(fin_res)
        
        loss_PC_l = []
        
        if not fin_res:
            detPC = torch.tensor(0.0).cuda()

        for di,g in fin_res.items():
            if g['TP'] is not None and g['FP'] is not None:
                TP_l = g['conf_scr'][:g['TP']]
                FP_l = g['conf_scr'][g['TP']:g['FP']+g['TP']]
                m = nn.Tanh()

                AC = TP_l[TP_l>=0.5] * m(TP_l[TP_l>=0.5])
                AN = TP_l[TP_l<0.5] * (1-m(TP_l[TP_l<0.5]))
                IC = (1-FP_l[FP_l>=0.5]) * m(FP_l[FP_l>=0.5])
                IN = (1-FP_l[FP_l<0.5]) * (1-m(FP_l[FP_l<0.5]))

                nAC = AC.sum()
                nAN = AN.sum()
                nIC = IC.sum()
                nIN = IN.sum()

                numr = nAN+nIC
                denom = nAC+nIN


                if denom>0.0:
                    detPC = torch.log(1+(numr.cuda()/denom.cuda()))
                    loss_PC_l.append(detPC)
                else:
                    detPC = torch.tensor(0.0).cuda()

            else:
                detPC = torch.tensor(0.0).cuda()
        
        if loss_PC_l:
            detPC_mean = torch.mean(torch.stack(loss_PC_l))
        else:
            detPC_mean = torch.tensor(0.0).cuda()
        
        return detPC_mean
        
        

        
class FastRCNNCalitLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        #print(input_shape)
        super(FastRCNNCalitLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals, gt_im_id, targets):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        #print(proposals)# normal no extra steps
        pred_instances, _ = self.inference(predictions, proposals)
        #print("inference",pred_instances)
        #exit()
        losses={}
        
        cali_loss = CalibrationLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            gt_im_id,
            pred_instances,
            targets,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()
        
        det_losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()
        for key in cali_loss.keys():
            losses[key] = cali_loss[key]
        for key in det_losses.keys():
            losses['cali_'+key] = det_losses[key]
        #print(cali_loss, det_losses)
        return losses

