import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from torch.amp import GradScaler
import torch.nn.functional as F
from torchvision.ops import batched_nms, roi_align, clip_boxes_to_image, box_iou


# Backbone ResNet18 using pretrained model
class Feature_Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = timm.create_model(model_name='resnet18.a1_in1k', pretrained=True)
        self.extractor.global_pool = nn.Identity()
        self.extractor.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor.forward_features(x)


class FasterRCNN_model(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 9
        self.stride = 32
        self.backbone = Feature_Extractor()

        ####################### Head RPN #######################################################
        # Phân loại objectness (foreground vs background) ứng với 9 anchors trên mỗi vùng feature map
        self.classifier_head = nn.Conv2d(
            in_channels=512,
            out_channels=2 * self.num_anchors,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Dự đoán độ lệch bounding box (dx, dy, dw, dh) cho 9 anchors trên mỗi vùng feature map
        self.Regression_bbx_head = nn.Conv2d(
            in_channels=512,
            out_channels=4 * self.num_anchors,
            kernel_size=3,
            stride=1,
            padding=1
        )
        ########################################################################################

        ################################### Head Fast RCNN #####################################
        self.classifier_head_fc1 = nn.Linear(
            in_features=512 * 7 * 7,
            out_features=1024,
            bias=True
        )
        self.classifier_head_fc2 = nn.Linear(
            in_features=1024,
            out_features=self.num_classes + 1,
            bias=True
        )

        self.regression_head_fc1 = nn.Linear(
            in_features=512 * 7 * 7,
            out_features=1024,
            bias=True
        )
        self.regression_head_fc2 = nn.Linear(
            in_features=1024,
            out_features=4,
            bias=True
        )
        ########################################################################################

    def get_proposal(
        self,
        proposal_delta: torch.Tensor,
        objectness_score: torch.Tensor,
        img_size: tuple[int, int],
        top_k: int = 1000,
        pre_nms_top_k: int = 2000,
        iou_threshold: float = 0.7
    ) -> list[torch.Tensor]:
        """
        Sinh ra danh sách các proposal boxes từ RPN predictions sau khi decode, clip và NMS.
        
        Args:
            proposal_delta: Tensor độ lệch do RPN dự đoán, shape (B, 4 * num_anchors, Oh, Ow) hoặc (B, M, 4).
            objectness_score: Tensor điểm objectness do RPN dự đoán, shape (B, 2 * num_anchors, Oh, Ow) hoặc (B, M, 2).
            img_size: Kích thước ảnh gốc (H, W).
            top_k: Số lượng proposal tối đa giữ lại sau NMS cho mỗi ảnh.
            pre_nms_top_k: Số lượng proposal điểm cao nhất lấy trước khi chạy NMS.
            iou_threshold: Ngưỡng IoU cho NMS.

        Returns:
            batch_proposals: Danh sách gồm B tensor, mỗi tensor chứa các proposal [x1, y1, x2, y2] của ảnh đó.
        """
        B = proposal_delta.shape[0]
        device = proposal_delta.device

        # Reshape delta nếu là 4D tensor từ Conv2d
        if proposal_delta.dim() == 4:
            Oh, Ow = proposal_delta.shape[2], proposal_delta.shape[3]
            deltas = proposal_delta.view(B, self.num_anchors, 4, Oh, Ow).permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
        else:
            deltas = proposal_delta
            Oh = int((proposal_delta.shape[1] // self.num_anchors) ** 0.5)
            Ow = Oh

        # Reshape objectness score nếu là 4D tensor từ Conv2d
        if objectness_score.dim() == 4:
            scores = objectness_score.view(B, self.num_anchors, 2, Oh, Ow).permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 2)
        else:
            scores = objectness_score

        # Điểm xác suất foreground (lớp 1)
        obj_scores = F.softmax(scores, dim=-1)[:, :, 1]  # (B, M)

        # Tạo anchors
        anchors = self.make_anchors(Oh, Ow, batch_size=B, stride=self.stride, device=device)  # (B, M, 4)

        # Decode delta kết hợp anchors -> proposals [xmin, ymin, xmax, ymax]
        proposals = self.decode_delta(anchors, deltas)  # (B, M, 4)
        proposals = clip_boxes_to_image(proposals, img_size)

        batch_proposals = []
        for b in range(B):
            b_boxes = proposals[b]
            b_scores = obj_scores[b]

            # Loại bỏ các box kích thước quá nhỏ
            ws = b_boxes[:, 2] - b_boxes[:, 0]
            hs = b_boxes[:, 3] - b_boxes[:, 1]
            valid_size = (ws > 1.0) & (hs > 1.0)
            b_boxes = b_boxes[valid_size]
            b_scores = b_scores[valid_size]

            if b_boxes.shape[0] == 0:
                fallback = torch.tensor([[0.0, 0.0, float(img_size[1]), float(img_size[0])]], device=device)
                batch_proposals.append(fallback)
                continue

            # Lấy top K điểm cao nhất trước NMS
            k = min(len(b_scores), pre_nms_top_k)
            topk_scores, topk_idx = torch.topk(b_scores, k, dim=-1)
            topk_boxes = b_boxes[topk_idx]

            # Áp dụng NMS
            keep = batched_nms(
                boxes=topk_boxes,
                scores=topk_scores,
                idxs=torch.zeros_like(topk_scores, dtype=torch.int64),
                iou_threshold=iou_threshold
            )
            keep = keep[:top_k]
            batch_proposals.append(topk_boxes[keep].detach())

        return batch_proposals

    def forward_fast_rcnn(
        self,
        FeatureMaps: torch.Tensor,
        proposals: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Thực hiện RoI Align trên Feature Maps và đưa qua 2 head của Fast R-CNN:
        - classifier_head: Phân loại đa lớp (C + 1 lớp, lớp 0 là background)
        - regression_head: Dự đoán độ lệch hộp giới hạn (4 tọa độ)
        """
        ROIPooling = roi_align(
            input=FeatureMaps,
            boxes=proposals,
            output_size=(7, 7),
            spatial_scale=1.0 / self.stride
        )
        ROIPooling = ROIPooling.view(ROIPooling.shape[0], -1)
        ROIPooling = F.dropout(ROIPooling, p=0.5, training=self.training)

        Fclassifier_fc1 = F.relu(self.classifier_head_fc1(ROIPooling))
        logits = self.classifier_head_fc2(Fclassifier_fc1)  # Shape: (Total_RoIs, num_classes + 1)

        FRegressor_fc1 = F.relu(self.regression_head_fc1(ROIPooling))
        box_deltas = self.regression_head_fc2(FRegressor_fc1)  # Shape: (Total_RoIs, 4)

        return logits, box_deltas

    def forward(
        self,
        x: torch.Tensor,
        proposals: list[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward end-to-end qua cả RPN và Fast R-CNN.
        Nếu proposals là None, RPN sẽ tự sinh ra proposals thông qua get_proposal.
        """
        img_size = (x.shape[2], x.shape[3])

        FeatureMaps = self.backbone(x)  # Shape: (B, 512, H_feat, W_feat)
        Proposal_delta = self.Regression_bbx_head(FeatureMaps)  # Shape: (B, 9 * 4, H_feat, W_feat)
        objectness_score = self.classifier_head(FeatureMaps)    # Shape: (B, 9 * 2, H_feat, W_feat)

        if proposals is None:
            proposals = self.get_proposal(Proposal_delta, objectness_score, img_size)

        logits, box_deltas = self.forward_fast_rcnn(FeatureMaps, proposals)

        return Proposal_delta, objectness_score, logits, box_deltas

    def make_anchors(
        self,
        O_h: int,
        O_w: int,
        batch_size: int = 1,
        stride: int = 32,
        scales: list = None,
        aspect_ratios: list = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Xác định tọa độ các anchors trên từng vùng (i, j) của feature map.

        1. Xác định tọa độ tâm (center_x, center_y) trên từng vùng (i, j) của feature map
           chiếu về tọa độ ảnh gốc theo stride.
        2. Đặt 9 anchors tại mỗi tâm với:
           - 3 scales mặc định: 128^2, 256^2, 512^2 (cạnh s tương ứng 128, 256, 512)
           - 3 aspect ratios (W:H) mặc định: 1:1, 1:2, 2:1 (r = [1.0, 0.5, 2.0])
           - w = s * sqrt(r), h = s / sqrt(r)
        3. Trả về thông tin của anchors gồm tọa độ tâm và kích thước [x_center, y_center, w, h] dưới dạng tensor.

        Args:
            O_h: Chiều cao của feature map.
            O_w: Chiều rộng của feature map.
            batch_size: Số lượng ảnh trong batch.
            stride: Stride của feature extractor so với ảnh gốc (mặc định 32 cho ResNet18).
            scales: Danh sách các giá trị scale (cạnh s sao cho diện tích s^2). Mặc định: [128.0, 256.0, 512.0].
            aspect_ratios: Danh sách tỉ lệ khung hình (w/h). Mặc định: [1.0, 0.5, 2.0] tương ứng 1:1, 1:2, 2:1.
            device: Thiết bị tính toán (CPU hoặc CUDA).

        Returns:
            anchors: Tensor tọa độ [x_center, y_center, w, h], shape (B, M, 4) với M = O_h * O_w * 9.
        """
        if scales is None:
            scales = [128.0, 256.0, 512.0]
        if aspect_ratios is None:
            aspect_ratios = [1.0, 0.5, 2.0]

        # Diện tích của anchor ở cùng 1 scale là bằng nhau 
        # 1. Tính kích thước (w, h) cho 9 anchors từ 3 scales và 3 aspect ratios
        # w * h = s^2 và w / h = r  =>  w = s * sqrt(r), h = s / sqrt(r)
        base_wh = []
        for s in scales:
            for r in aspect_ratios:
                w = s * (r ** 0.5)
                h = s / (r ** 0.5)
                base_wh.append([w, h])
        base_wh = torch.tensor(base_wh, dtype=torch.float32, device=device)  # shape (9, 2)
        num_anchors = base_wh.shape[0]

        # 2. Xác định tọa độ tâm (center_x, center_y) cho từng vùng (i, j) của feature map
        # Mỗi cell (i, j) trên feature map ứng với vùng trên ảnh gốc có tâm ở giữa cell: (index + 0.5) * stride
        ys = (torch.arange(O_h, dtype=torch.float32, device=device) + 0.5) * stride
        xs = (torch.arange(O_w, dtype=torch.float32, device=device) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # Shape: (O_h, O_w)

        # 3. Kết hợp tâm với 9 anchors tại từng cell (i, j)
        grid_x = grid_x.unsqueeze(-1).repeat(1, 1, num_anchors)  # (O_h, O_w, 9)
        grid_y = grid_y.unsqueeze(-1).repeat(1, 1, num_anchors)  # (O_h, O_w, 9)

        ws = base_wh[:, 0].view(1, 1, num_anchors).expand(O_h, O_w, num_anchors)  # (O_h, O_w, 9)
        hs = base_wh[:, 1].view(1, 1, num_anchors).expand(O_h, O_w, num_anchors)  # (O_h, O_w, 9)

        # anchors shape: (O_h, O_w, 9, 4) chứa [x_center, y_center, w, h]
        anchors = torch.stack([grid_x, grid_y, ws, hs], dim=-1)

        # Flatten về (M, 4) với M = O_h * O_w * 9
        anchors = anchors.view(-1, 4)

        # Mở rộng theo batch_size -> (B, M, 4)
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        return anchors

    def encode_delta(self, anchor: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
        """
        Tính toán độ lệch (delta) giữa anchor [x_center, y_center, w, h] và ground truth box [xmin, ymin, xmax, ymax].
        dx = (x_gt - x_a) / w_a
        dy = (y_gt - y_a) / h_a
        dw = log(w_gt / w_a)
        dh = log(h_gt / h_a)
        """
        xcenter_anchor = anchor[..., 0]
        ycenter_anchor = anchor[..., 1]
        w_anchor = anchor[..., 2]
        h_anchor = anchor[..., 3]

        w_gt = gt_box[..., 2] - gt_box[..., 0]
        h_gt = gt_box[..., 3] - gt_box[..., 1]
        xcenter_gt = gt_box[..., 0] + 0.5 * w_gt
        ycenter_gt = gt_box[..., 1] + 0.5 * h_gt

        # Giữ độ ổn định số học tránh chia cho 0 hoặc log số âm
        w_anchor = torch.clamp(w_anchor, min=1e-6)
        h_anchor = torch.clamp(h_anchor, min=1e-6)
        w_gt = torch.clamp(w_gt, min=1e-6)
        h_gt = torch.clamp(h_gt, min=1e-6)

        dx = (xcenter_gt - xcenter_anchor) / w_anchor
        dy = (ycenter_gt - ycenter_anchor) / h_anchor
        dw = torch.log(w_gt / w_anchor)
        dh = torch.log(h_gt / h_anchor)

        return torch.stack([dx, dy, dw, dh], dim=-1)

    def decode_delta(self, anchor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Giải mã độ lệch (delta) do mô hình dự đoán kết hợp với anchor boxes [x_center, y_center, w, h]
        để trả về tọa độ bounding box thực tế [xmin, ymin, xmax, ymax].
        """
        xcenter_anchor = anchor[..., 0]
        ycenter_anchor = anchor[..., 1]
        w_anchor = anchor[..., 2]
        h_anchor = anchor[..., 3]

        dx = delta[..., 0]
        dy = delta[..., 1]
        dw = torch.clamp(delta[..., 2], min=-10.0, max=10.0)
        dh = torch.clamp(delta[..., 3], min=-10.0, max=10.0)

        xcenter_pred = xcenter_anchor + dx * w_anchor
        ycenter_pred = ycenter_anchor + dy * h_anchor
        w_pred = w_anchor * torch.exp(dw)
        h_pred = h_anchor * torch.exp(dh)

        xmin = xcenter_pred - 0.5 * w_pred
        ymin = ycenter_pred - 0.5 * h_pred
        xmax = xcenter_pred + 0.5 * w_pred
        ymax = ycenter_pred + 0.5 * h_pred

        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def build_target(
        self,
        anchors: torch.Tensor,
        GT_boxes: torch.Tensor,
        GT_labels: torch.Tensor,
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.3,
        mask_GT_boxes: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Xác định target labels và target deltas cho các anchors phục vụ huấn luyện 2 head của RPN.
        Target label cho RPN:
            - 1: Foreground (chứa object)
            - 0: Background
            - -1: Ignore (bỏ qua khi tính loss)
        """
        B, M, _ = anchors.shape
        device = anchors.device

        labels = torch.zeros((B, M), dtype=torch.int64, device=device)
        target_delta_bbx = torch.zeros((B, M, 4), dtype=torch.float32, device=device)

        # Chuyển anchors từ [x_center, y_center, w, h] sang [xmin, ymin, xmax, ymax]
        anchor_x1 = anchors[..., 0] - 0.5 * anchors[..., 2]
        anchor_y1 = anchors[..., 1] - 0.5 * anchors[..., 3]
        anchor_x2 = anchors[..., 0] + 0.5 * anchors[..., 2]
        anchor_y2 = anchors[..., 1] + 0.5 * anchors[..., 3]
        anchors_corners = torch.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], dim=-1)

        area_anchors = (anchors[..., 2] * anchors[..., 3]).clamp(min=1e-6)
        area_gts = ((GT_boxes[..., 2] - GT_boxes[..., 0]).clamp(min=0) * (GT_boxes[..., 3] - GT_boxes[..., 1]).clamp(min=0))

        left_top = torch.max(anchors_corners[:, :, None, :2], GT_boxes[:, None, :, :2])
        right_bottom = torch.min(anchors_corners[:, :, None, 2:], GT_boxes[:, None, :, 2:])

        # Tính diện tích phần giao nhau giữa anchor và GT
        intersection = torch.clamp(right_bottom - left_top, min=0)  # (B, M, N, 2)
        intersection = intersection[:, :, :, 0] * intersection[:, :, :, 1]  # (B, M, N)

        union = area_anchors[:, :, None] + area_gts[:, None, :] - intersection
        iou = intersection / torch.clamp(union, min=1e-6)  # (B, M, N)

        if mask_GT_boxes is not None:
            iou = iou * mask_GT_boxes.unsqueeze(1).float()

        for b in range(B):
            iou_b = iou[b]
            iou_max, idx_best_gt = torch.max(iou_b, dim=1)
            pos_mask = (iou_max >= positive_threshold)

            # Đảm bảo mỗi GT box hợp lệ có ít nhất 1 anchor tốt nhất (highest IoU) được gán positive
            if mask_GT_boxes is not None:
                valid_gt_indices = torch.nonzero(mask_GT_boxes[b]).squeeze(1)
            else:
                valid_gt_indices = torch.arange(GT_boxes.shape[1], device=device)

            if valid_gt_indices.numel() > 0:
                gt_max_iou, gt_argmax = torch.max(iou_b[:, valid_gt_indices], dim=0)
                valid_matches = gt_max_iou > 0
                if valid_matches.sum() > 0:
                    best_anchor_indices = gt_argmax[valid_matches]
                    pos_mask[best_anchor_indices] = True
                    idx_best_gt[best_anchor_indices] = valid_gt_indices[valid_matches]

            ignore_mask = (iou_max < positive_threshold) & (iou_max >= negative_threshold) & (~pos_mask)

            labels[b, pos_mask] = 1
            labels[b, ignore_mask] = -1

            if pos_mask.sum() > 0:
                matched_gt_boxes = GT_boxes[b, idx_best_gt[pos_mask]]
                target_delta_bbx[b, pos_mask] = self.encode_delta(anchors[b, pos_mask], matched_gt_boxes)

        return labels, target_delta_bbx

    def build_roi_targets(
        self,
        proposals: list[torch.Tensor],
        GT_boxes: torch.Tensor,
        GT_labels: torch.Tensor,
        mask_GT_boxes: torch.Tensor = None,
        num_samples: int = 128,
        pos_fraction: float = 0.25,
        pos_threshold: float = 0.5,
        neg_threshold_hi: float = 0.5,
        neg_threshold_lo: float = 0.0
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Ghép các proposals với Ground Truth để lấy mẫu (sampling) và tạo target cho 2 head của Fast R-CNN:
        - Thêm GT boxes vào tập proposal ứng viên để đảm bảo có positive RoIs.
        - Tính IoU giữa RoIs và GT boxes:
            + IoU >= pos_threshold: gán nhãn class GT tương ứng (1..C), tính delta target.
            + neg_threshold_lo <= IoU < neg_threshold_hi: gán nhãn 0 (background).
        - Lấy mẫu cân bằng theo pos_fraction (mặc định 25% positive, 75% background).

        Returns:
            sampled_rois_list: Danh sách RoIs đã lấy mẫu cho từng ảnh trong batch.
            roi_labels: Tensor nhãn lớp mục tiêu của tất cả RoIs (Shape: N_sampled,).
            roi_target_deltas: Tensor độ lệch mục tiêu của tất cả RoIs (Shape: N_sampled, 4).
        """
        B = len(proposals)
        device = proposals[0].device
        sampled_rois_list = []
        roi_labels_list = []
        roi_targets_list = []

        for b in range(B):
            if mask_GT_boxes is not None:
                valid_gt_mask = mask_GT_boxes[b]
            else:
                valid_gt_mask = torch.ones(GT_boxes.shape[1], dtype=torch.bool, device=device)

            valid_gt_boxes = GT_boxes[b][valid_gt_mask]
            valid_gt_labels = GT_labels[b][valid_gt_mask]
            prop_b = proposals[b]

            # Nếu ảnh không có GT box nào
            if valid_gt_boxes.shape[0] == 0:
                k = min(num_samples, prop_b.shape[0])
                sampled_rois_list.append(prop_b[:k])
                roi_labels_list.append(torch.zeros(k, dtype=torch.int64, device=device))
                roi_targets_list.append(torch.zeros((k, 4), dtype=torch.float32, device=device))
                continue

            # Nối GT boxes vào candidate proposals để bảo đảm luôn có RoIs chất lượng cao
            all_rois = torch.cat([valid_gt_boxes, prop_b], dim=0)
            iou = box_iou(all_rois, valid_gt_boxes)
            max_iou, best_gt_idx = torch.max(iou, dim=1)

            pos_mask = (max_iou >= pos_threshold)
            pos_idx = torch.nonzero(pos_mask).squeeze(1)

            neg_mask = (max_iou < neg_threshold_hi) & (max_iou >= neg_threshold_lo)
            neg_idx = torch.nonzero(neg_mask).squeeze(1)

            pos_max = int(num_samples * pos_fraction)
            num_pos = min(pos_idx.numel(), pos_max)
            if pos_idx.numel() > 0:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:num_pos]]

            num_neg = min(neg_idx.numel(), num_samples - num_pos)
            if neg_idx.numel() > 0:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:num_neg]]

            sampled_idx = torch.cat([pos_idx, neg_idx])
            sampled_rois = all_rois[sampled_idx]

            sampled_labels = torch.zeros(len(sampled_idx), dtype=torch.int64, device=device)
            if num_pos > 0:
                sampled_labels[:num_pos] = valid_gt_labels[best_gt_idx[pos_idx]]

            sampled_targets = torch.zeros((len(sampled_idx), 4), dtype=torch.float32, device=device)
            if num_pos > 0:
                pos_rois = sampled_rois[:num_pos]
                matched_gt = valid_gt_boxes[best_gt_idx[pos_idx]]
                w_roi = (pos_rois[:, 2] - pos_rois[:, 0]).clamp(min=1e-6)
                h_roi = (pos_rois[:, 3] - pos_rois[:, 1]).clamp(min=1e-6)
                cx_roi = pos_rois[:, 0] + 0.5 * w_roi
                cy_roi = pos_rois[:, 1] + 0.5 * h_roi
                rois_cxcywh = torch.stack([cx_roi, cy_roi, w_roi, h_roi], dim=-1)
                sampled_targets[:num_pos] = self.encode_delta(rois_cxcywh, matched_gt)

            sampled_rois_list.append(sampled_rois)
            roi_labels_list.append(sampled_labels)
            roi_targets_list.append(sampled_targets)

        return sampled_rois_list, torch.cat(roi_labels_list, dim=0), torch.cat(roi_targets_list, dim=0)

    def NMS(
        self,
        logits: torch.Tensor,
        delta_boxes: torch.Tensor,
        iou_threshold: float = 0.6,
        confidence_threshold: float = 0.05
    ) -> list:
        """
        Non-Maximum Suppression (NMS) để chọn lọc các region proposals dự đoán cho từng ảnh trong batch.

        Args:
            logits: Output logits từ classifier_head.
            delta_boxes: Output delta từ Regression_bbx_head.
            iou_threshold: Ngưỡng IoU cho NMS.
            confidence_threshold: Ngưỡng độ tin cậy tối thiểu.

        Returns:
            results: Danh sách kết quả [{'boxes': ..., 'scores': ..., 'labels': ...}] cho từng ảnh trong batch.
        """
        B = logits.shape[0]
        C = self.num_classes
        num_anchors = self.num_anchors

        if logits.dim() == 4:
            H, W = logits.shape[2], logits.shape[3]
            logits = logits.view(B, num_anchors, C, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, -1, C)
            delta_boxes = delta_boxes.view(B, num_anchors, 4, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
            anchors = self.make_anchors(H, W, batch_size=B, stride=self.stride, device=logits.device)
        else:
            H = int((logits.shape[1] // num_anchors) ** 0.5)
            W = H
            anchors = self.make_anchors(H, W, batch_size=B, stride=self.stride, device=logits.device)

        pred_boxes = self.decode_delta(anchors, delta_boxes)  # (B, M, 4)
        pred_scores = F.softmax(logits, dim=-1)                # (B, M, C)

        results = []
        for b in range(B):
            b_boxes = pred_boxes[b]
            b_scores = pred_scores[b]

            if C > 1:
                obj_scores, class_ids = torch.max(b_scores[:, 1:], dim=-1)
                class_ids = class_ids + 1
            else:
                obj_scores = b_scores[:, 0]
                class_ids = torch.zeros_like(obj_scores, dtype=torch.int64)

            keep_mask = obj_scores >= confidence_threshold
            filt_boxes = b_boxes[keep_mask]
            filt_scores = obj_scores[keep_mask]
            filt_classes = class_ids[keep_mask]

            if filt_boxes.shape[0] == 0:
                results.append({
                    'boxes': torch.empty((0, 4), device=logits.device),
                    'scores': torch.empty((0,), device=logits.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=logits.device)
                })
                continue

            keep = batched_nms(filt_boxes, filt_scores, filt_classes, iou_threshold=iou_threshold)
            results.append({
                'boxes': filt_boxes[keep],
                'scores': filt_scores[keep],
                'labels': filt_classes[keep]
            })
        return results


class TrainModel(nn.Module):
    """
    Class quản lý và thực thi quá trình huấn luyện End-to-End mô hình Faster R-CNN.
    Tổng hàm Loss được tính từ 4 loss của 4 head:
        1. RPN Classification Loss: Phân loại nhãn objectness (foreground / background) cho các anchors.
        2. RPN Bounding Box Regression Loss: Hồi quy tọa độ bounding box cho các positive anchors.
        3. Fast R-CNN Classification Loss: Phân loại đa lớp cho các RoIs được lấy mẫu.
        4. Fast R-CNN Bounding Box Regression Loss: Hồi quy tọa độ bounding box cho các positive RoIs.
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Cấu hình Optimizer tối ưu toàn bộ các module trong mô hình End-to-End
        self.optimizer = torch.optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': learning_rate / 10, 'weight_decay': 0.0005},
            {'params': self.model.classifier_head.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.Regression_bbx_head.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.classifier_head_fc1.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.classifier_head_fc2.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.regression_head_fc1.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
            {'params': self.model.regression_head_fc2.parameters(), 'lr': learning_rate, 'weight_decay': 0.0005},
        ])

        # 4 Hàm loss cho 4 head của mô hình
        # 1. Head 1 (RPN Classifier): Phân loại nhị phân objectness (0: bg, 1: fg, -1: ignore)
        self.rpn_classifier_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        # 2. Head 2 (RPN Regressor): SmoothL1Loss cho bounding box delta của positive anchors
        self.rpn_regression_loss = nn.SmoothL1Loss(reduction='sum')
        # 3. Head 3 (Fast R-CNN Classifier): CrossEntropy cho phân loại đối tượng (lớp 0 là background)
        self.fast_rcnn_classifier_loss = nn.CrossEntropyLoss(reduction='sum')
        # 4. Head 4 (Fast R-CNN Regressor): SmoothL1Loss cho bounding box delta của positive RoIs
        self.fast_rcnn_regression_loss = nn.SmoothL1Loss(reduction='sum')

        # Tương thích ngược với các tên thuộc tính cũ nếu có
        self.classifier_loss = self.rpn_classifier_loss
        self.regression_loss = self.rpn_regression_loss

        self.n_epochs = n_epochs
        self.dataloader = dataloader
        self.batch_size = batch_size

        # Lưu lịch sử quá trình huấn luyện
        self.TrainLosses = []
        self.RPN_Losses = []
        self.FastRCNN_Losses = []
        self.TrainAcc = []
        self.history = {
            'total_loss': [],
            'rpn_cls_loss': [],
            'rpn_reg_loss': [],
            'fast_cls_loss': [],
            'fast_reg_loss': [],
            'rpn_acc': [],
            'fast_acc': []
        }

    def get_accuracy(self, logits: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Tính độ chính xác phân loại của RPN trên các anchors hợp lệ (loại trừ các vị trí ignore -1).
        """
        mask = (y_true != -1)
        if mask.sum() == 0:
            return 0.0
        y_pred = torch.argmax(logits, dim=-1)
        correct_tokens = (y_pred == y_true) & mask
        return (correct_tokens.sum() / mask.sum()).item()

    def get_fast_rcnn_accuracy(self, roi_logits: torch.Tensor, roi_labels: torch.Tensor) -> float:
        """
        Tính độ chính xác phân loại của Fast R-CNN trên các RoIs đã lấy mẫu.
        """
        if roi_labels.numel() == 0:
            return 0.0
        y_pred = torch.argmax(roi_logits, dim=-1)
        correct = (y_pred == roi_labels).sum().item()
        return correct / roi_labels.numel()

    def forward(self) -> None:
        """
        Huấn luyện mô hình Faster R-CNN qua n_epochs với Mixed Precision (AMP).
        Loss tổng là tổng của 4 hàm loss:
            Loss_total = Loss_RPN_cls + Loss_RPN_reg + Loss_FastRCNN_cls + Loss_FastRCNN_reg
        """
        self.TrainLosses = []
        self.RPN_Losses = []
        self.FastRCNN_Losses = []
        self.TrainAcc = []

        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.model.train()
            total_loss_epoch = 0.0
            rpn_cls_loss_epoch = 0.0
            rpn_reg_loss_epoch = 0.0
            fast_cls_loss_epoch = 0.0
            fast_reg_loss_epoch = 0.0
            rpn_acc_epoch = 0.0
            fast_acc_epoch = 0.0
            total_batches = 0

            for batch in self.dataloader:
                imgs = batch[0].to(self.device)
                gt_boxes = batch[1].to(self.device)
                gt_labels = batch[2].to(self.device)
                mask_gt_boxes = batch[4].to(self.device) if len(batch) > 4 else None

                B = imgs.shape[0]
                img_size = (imgs.shape[2], imgs.shape[3])

                with torch.autocast(device_type='cuda' if 'cuda' in str(self.device) else 'cpu'):
                    # 1. Trích xuất đặc trưng từ Backbone
                    feature_maps = self.model.backbone(imgs)  # Shape: (B, 512, Oh, Ow)
                    Oh, Ow = feature_maps.shape[2], feature_maps.shape[3]
                    num_anchors = self.model.num_anchors
                    M = Oh * Ow * num_anchors

                    # 2. Forward qua 2 head của RPN
                    rpn_deltas_raw = self.model.Regression_bbx_head(feature_maps)  # (B, 4 * num_anchors, Oh, Ow)
                    rpn_scores_raw = self.model.classifier_head(feature_maps)      # (B, 2 * num_anchors, Oh, Ow)

                    # Reshape outputs RPN khớp với thứ tự anchors: (B, M, 4) và (B, M, 2)
                    rpn_deltas = rpn_deltas_raw.view(B, num_anchors, 4, Oh, Ow).permute(0, 3, 4, 1, 2).contiguous().view(B, M, 4)
                    rpn_scores = rpn_scores_raw.view(B, num_anchors, 2, Oh, Ow).permute(0, 3, 4, 1, 2).contiguous().view(B, M, 2)

                    # 3. Tạo anchors và targets cho RPN
                    anchors = self.model.make_anchors(
                        O_h=Oh,
                        O_w=Ow,
                        batch_size=B,
                        stride=self.model.stride,
                        device=self.device
                    )

                    rpn_target_labels, rpn_target_deltas = self.model.build_target(
                        anchors=anchors,
                        GT_boxes=gt_boxes,
                        GT_labels=gt_labels,
                        positive_threshold=0.7,
                        negative_threshold=0.3,
                        mask_GT_boxes=mask_gt_boxes
                    )

                    # 4. Tính 2 Loss cho RPN:
                    # Loss 1: RPN Classification Loss (Objectness)
                    valid_rpn = (rpn_target_labels != -1)
                    num_valid_rpn = valid_rpn.sum().item()
                    rpn_cls_loss = self.rpn_classifier_loss(
                        rpn_scores.view(-1, 2),
                        rpn_target_labels.view(-1)
                    ) / max(1, num_valid_rpn)

                    # Loss 2: RPN Bounding Box Regression Loss
                    pos_rpn = (rpn_target_labels > 0)
                    num_pos_rpn = pos_rpn.sum().item()
                    if num_pos_rpn > 0:
                        rpn_reg_loss = self.rpn_regression_loss(
                            rpn_deltas[pos_rpn],
                            rpn_target_deltas[pos_rpn]
                        ) / max(1, num_pos_rpn)
                    else:
                        rpn_reg_loss = torch.tensor(0.0, device=self.device)

                    # 5. Sinh RoI Proposals từ RPN
                    proposals = self.model.get_proposal(
                        rpn_deltas_raw,
                        rpn_scores_raw,
                        img_size=img_size,
                        top_k=1000
                    )

                    # 6. Gán targets và lấy mẫu (sampling) RoIs cho Fast R-CNN
                    sampled_rois, roi_labels, roi_target_deltas = self.model.build_roi_targets(
                        proposals=proposals,
                        GT_boxes=gt_boxes,
                        GT_labels=gt_labels,
                        mask_GT_boxes=mask_gt_boxes,
                        num_samples=128,
                        pos_fraction=0.25
                    )

                    # 7. Forward qua 2 head của Fast R-CNN (RoI Align + FC layers)
                    roi_logits, roi_box_deltas = self.model.forward_fast_rcnn(
                        feature_maps,
                        sampled_rois
                    )

                    # 8. Tính 2 Loss cho Fast R-CNN:
                    # Loss 3: Fast R-CNN Classification Loss (Đa lớp: 0..C)
                    num_rois = roi_labels.shape[0]
                    fast_rcnn_cls_loss = self.fast_rcnn_classifier_loss(
                        roi_logits,
                        roi_labels
                    ) / max(1, num_rois)

                    # Loss 4: Fast R-CNN Bounding Box Regression Loss (Chỉ tính trên positive RoIs)
                    pos_roi = (roi_labels > 0)
                    num_pos_roi = pos_roi.sum().item()
                    if num_pos_roi > 0:
                        fast_rcnn_reg_loss = self.fast_rcnn_regression_loss(
                            roi_box_deltas[pos_roi],
                            roi_target_deltas[pos_roi]
                        ) / max(1, num_pos_roi)
                    else:
                        fast_rcnn_reg_loss = torch.tensor(0.0, device=self.device)

                    # 9. TỔNG CỦA 4 HÀM LOSS (RPN cls + RPN reg + Fast R-CNN cls + Fast R-CNN reg)
                    total_loss = rpn_cls_loss + rpn_reg_loss + fast_rcnn_cls_loss + fast_rcnn_reg_loss

                    # Tính accuracy
                    rpn_acc = self.get_accuracy(rpn_scores, rpn_target_labels)
                    fast_rcnn_acc = self.get_fast_rcnn_accuracy(roi_logits, roi_labels)

                # 10. Backward và cập nhật trọng số
                self.optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                total_loss_epoch += total_loss.item()
                rpn_cls_loss_epoch += rpn_cls_loss.item()
                rpn_reg_loss_epoch += rpn_reg_loss.item()
                fast_cls_loss_epoch += fast_rcnn_cls_loss.item()
                fast_reg_loss_epoch += fast_rcnn_reg_loss.item()
                rpn_acc_epoch += rpn_acc
                fast_acc_epoch += fast_rcnn_acc
                total_batches += 1

            epoch_loss = total_loss_epoch / max(1, total_batches)
            epoch_rpn_cls = rpn_cls_loss_epoch / max(1, total_batches)
            epoch_rpn_reg = rpn_reg_loss_epoch / max(1, total_batches)
            epoch_fast_cls = fast_cls_loss_epoch / max(1, total_batches)
            epoch_fast_reg = fast_reg_loss_epoch / max(1, total_batches)
            epoch_rpn_acc = rpn_acc_epoch / max(1, total_batches)
            epoch_fast_acc = fast_acc_epoch / max(1, total_batches)

            self.TrainLosses.append(epoch_loss)
            self.RPN_Losses.append(epoch_rpn_cls + epoch_rpn_reg)
            self.FastRCNN_Losses.append(epoch_fast_cls + epoch_fast_reg)
            self.TrainAcc.append(epoch_fast_acc)

            self.history['total_loss'].append(epoch_loss)
            self.history['rpn_cls_loss'].append(epoch_rpn_cls)
            self.history['rpn_reg_loss'].append(epoch_rpn_reg)
            self.history['fast_cls_loss'].append(epoch_fast_cls)
            self.history['fast_reg_loss'].append(epoch_fast_reg)
            self.history['rpn_acc'].append(epoch_rpn_acc)
            self.history['fast_acc'].append(epoch_fast_acc)

            tqdm.write(
                f"Epoch [{epoch + 1}/{self.n_epochs}] - "
                f"Total Loss: {epoch_loss:.4f} | "
                f"RPN (cls: {epoch_rpn_cls:.4f}, reg: {epoch_rpn_reg:.4f}, acc: {epoch_rpn_acc * 100:.2f}%) | "
                f"Fast R-CNN (cls: {epoch_fast_cls:.4f}, reg: {epoch_fast_reg:.4f}, acc: {epoch_fast_acc * 100:.2f}%)"
            )
