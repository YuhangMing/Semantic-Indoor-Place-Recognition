More and Complete results in "/media/yohann/Datasets/Network_Logs/CGS-Net_paper_logs"

[Segmentation, Full SemanticKitti/ScanNet Dataset]
Log_2021-06-11_02-47-11: SemanticKitti with deformed kernels. 	epoch800-500.
Log_2021-06-11_04-20-14: SemanticKitti with rigid kernels.    	epoch800-500.
Log_2021-06-16_02-31-04: ScanNet SLAM without color features. 	epoch136+500-600.
Log_2021-06-16_02-42-30: ScanNet SLAM with color features.    	epoch119+500-600.

[VLAD, Full ScanNet Dataset, pcd selected at 2 FPS and at least 0.7m apart]
Following are trained without point cloud size limits
**Recog_Log_2021-07-02_03-51-36**:
	Adam, 5 feats, 6 neg samples, 50 epochs, 35000 step size, Triplet loss
	First 6 epochs merged from `Recog_Log_2021-06-27_13-47-12`
	Started at e06-i0
	Stopped at e022-i13149
**Recog_Log_2021-07-07_06-41-29**:
	Adam, 3 feats, 6 neg samples, 50 epochs, 35000 step size, Triplet loss
	epoch 0-5 merged from `Recog_Log_2021-06-27_13-59-08`
	epoch 6-17 merged from `Recog_Log_2021-07-01_07-48-10`
	Started at e18-i0
	Stopped at e022-i14453.
**Recog_Log_2021-07-01_07-55-26**:
	Adam, 5 feats, 6 neg samples, 50 epochs, 35000 step size, Quadruplet loss
	First 2 epochs merged from `Recog_Log_2021-06-29_12-43-48`
	Started at e02-i0
	Stopped at e020-i13977

Following are trained with point cloud size limits
[Checkpoints missing for the following trainings]
Recog_Log_2021-07-29_11-25-46:
	Adam, 5 feats, 6 neg samples, 25 epochs, 35000 step size, Quadruplet loss
	4096 fixed input pts number, no color
	Stopped at e021-i5244
	*checkpoints* refer to `PRNet_4096_PNVlad-Comp/results/Recog_Log_2021-07-29_11-25-46`
Recog_Log_2021-07-29_17-53-02
	Adam, 5 feats, 6 neg samples, 25 epochs, 35000 step size, Quadruplet loss
	no color
	Stopped at e020-i4117
Recog_Log_2021-08-17_23-32-30
	Adam, 5 feats, 6 neg samples, 25 epochs, 35000 step size, Quadruplet loss
	With color, trained from scratch
	`Full results refer to the scratch folder`

Recog_Log_2021-08-20_22-39-43
	Adam, 5 feats, 6 neg samples, 30 epochs, 35000 step size, Quadruplet loss
	same setup as `Recog_Log_2021-07-01_07-55-26` but with additional pcd size limits.
Recog_Log_2021-08-29_13-46-24
	Continue training from `Recog_Log_2021-08-20_22-39-43`.

