import datetime
import logging
import os


from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import cmd_args
import datasets
import transforms
from main_utils import *
from models_occlusion_kitti import ThreeDFlow_Kitti, multiScaleLoss


#for distributed
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def main(distributed=False, rank=0):

    if "NUMBA_DISABLE_JIT" in os.environ:
        del os.environ["NUMBA_DISABLE_JIT"]

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        args.gpu if args.multi_gpu is None else "0,1,2,3"
    )

    """CREATE DIR"""
    experiment_dir = Path("./experiment/")
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(
        str(experiment_dir)
        + "/ThreeDFlow%sFlyingthings3d-" % args.model_name
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    )
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)
    # os.system("cp %s %s" % ("models_occlusion.py", log_dir))
    # os.system("cp %s %s" % ("pointconv_util.py", log_dir))
    # os.system("cp %s %s" % ("train_occlusion.py", log_dir))
    # os.system("cp %s %s" % ("config_train_occlusion.yaml", log_dir))

    """LOG"""
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        str(log_dir) + "/train_%s_sceneflow.txt" % args.model_name
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        "----------------------------------------TRAINING----------------------------------"
    )
    logger.info("PARAMETER ...")
    logger.info(args)

    blue = lambda x: "\033[94m" + x + "\033[0m"
    model = ThreeDFlow_Kitti(args.is_training)

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Augmentation(
            args.aug_together, args.aug_pc2, args.data_process, args.num_points
        ),
        num_points=args.num_points,
        data_root=args.data_root,
        # full=args.full,
    )
    logger.info("train_dataset: " + str(train_dataset))
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if distributed else True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
        sampler=train_sampler,
    )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(
            args.data_process, args.num_points, args.allow_less_points
        ),
        num_points=args.num_points,
        data_root=args.data_root,
    )
    logger.info("val_dataset: " + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
        sampler=val_sampler,
    )

    """GPU selection and multi-GPU"""
    # if args.multi_gpu is not None:
    #     device_ids = [int(x) for x in args.multi_gpu.split(",")]
    #     torch.backends.cudnn.benchmark = True
    #     model.cuda(device_ids[0])
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    # else:
    #     model.cuda()
        
    if distributed:
        model.to(rank)
        model = DDP(model, device_ids=[rank])

    if args.pretrain is not None:
        """
        state_dict = torch.load(args.pretrain)
        from collections import OrderedDict

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            k = "module." + k
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print("load model %s" % args.pretrain)
        logger.info("load model %s" % args.pretrain)
        """

        model.load_state_dict(torch.load(args.pretrain))
        print("load model %s" % args.pretrain)
        logger.info("load model %s" % args.pretrain)
    else:
        print("Training from scratch")
        logger.info("Training from scratch")

    pretrain = args.pretrain
    init_epoch = 0  # int(pretrain[-14:-11]) if args.pretrain is not None else 0

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
        )

    optimizer.param_groups[0]["initial_lr"] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=80, gamma=0.5, last_epoch=init_epoch - 1
    )
    LEARNING_RATE_CLIP = 1e-5

    history = defaultdict(lambda: list())
    best_epe = 1000.0
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]["lr"], LEARNING_RATE_CLIP)
        print("Learning rate:%f" % lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(
            enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9
        ):
            pos1, pos2, norm1, norm2, flow, mask = data

            pos1 = pos1.cuda().float()
            pos2 = pos2.cuda().float()
            norm1 = norm1.cuda().float()
            norm2 = norm2.cuda().float()
            flow = flow.cuda().float()
            mask = mask.unsqueeze(2).cuda().float()

            model = model.train()

            pred_flows, gt_flows, pc1, pc2, _, _, _mask = model(
                pos1, pos2, norm1, norm2, flow, mask
            )

            loss = multiScaleLoss(pred_flows, gt_flows, _mask)

            history["loss"].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        if rank == 0:
            str_out = "EPOCH %d %s mean loss: %f" % (epoch, blue("train"), train_loss)
            print(str_out)
            logger.info(str_out)

        eval_epe3d, eval_loss = eval_sceneflow(model.eval(), val_loader)
        if rank == 0:
            str_out = "EPOCH %d %s mean epe3d: %f  mean eval loss: %f" % (
                epoch,
                blue("eval"),
                eval_epe3d,
                eval_loss,
            )
            print(str_out)
            logger.info(str_out)

            if eval_epe3d < best_epe:
                best_epe = eval_epe3d
                if args.multi_gpu is not None:
                    torch.save(
                        model.module.state_dict(),
                        "%s/%s_%.3d_%.4f.pth"
                        % (checkpoints_dir, args.model_name, epoch, best_epe),
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        "%s/%s_%.3d_%.4f.pth"
                        % (checkpoints_dir, args.model_name, epoch, best_epe),
                    )
                logger.info("Save model ...")
                print("Save model ...")
            print("Best epe loss is: %.5f" % (best_epe))
            logger.info("Best epe loss is: %.5f" % (best_epe))


def eval_sceneflow(model, loader, distributed=False, ngpus=1, rank=0):

    metrics = defaultdict(lambda: list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, mask = data

        pos1 = pos1.cuda().float()
        pos2 = pos2.cuda().float()
        norm1 = norm1.cuda().float()
        norm2 = norm2.cuda().float()
        flow = flow.cuda().float()
        mask = mask.unsqueeze(2).cuda().float()

        with torch.no_grad() and rank == 0:
            pred_flows, gt_flows, pc1, pc2, _, _, _mask = model(
                pos1, pos2, norm1, norm2, flow, mask
            )
            if distributed:
                pred_flows_gather_list = [torch.zeros_like(pred_flows) for _ in range(ngpus)]
                torch.distributed.all_gather(pred_flows_gather_list, pred_flows)
                
                gt_flows_gather_list = [torch.zeros_like(gt_flows) for _ in range(ngpus)]
                torch.distributed.all_gather(gt_flows_gather_list, gt_flows)
                
                mask_gather_list = [torch.zeros_like(_mask) for _ in range(ngpus)]
                torch.distributed.all_gather(mask_gather_list, _mask)
                # cat
                pred_flows_gather_list = torch.cat(pred_flows_gather_list)
                gt_flows_gather_list = torch.cat(gt_flows_gather_list)
                mask_gather_list = torch.cat(mask_gather_list)
                pred_flows, gt_flows, _mask = pred_flows_gather_list, gt_flows_gather_list, mask_gather_list

            eval_loss = multiScaleLoss(pred_flows, gt_flows, _mask)
            mask_sum = torch.sum(_mask[0], 1)  # B 1
            error = torch.norm(
                pred_flows[0].permute(0, 2, 1) - gt_flows[0].permute(0, 2, 1),
                dim=2,
                keepdim=True,
            )  # B 2048
            epe3d = (
                torch.sum(error * _mask[0], 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
            )
            epe3d = torch.mean(epe3d)

            metrics["epe3d_loss"].append(epe3d.cpu().data.numpy())
            metrics["eval_loss"].append(eval_loss.cpu().data.numpy())

    if rank==0:
        mean_epe3d = np.mean(metrics["epe3d_loss"])
        mean_eval = np.mean(metrics["eval_loss"])

    return mean_epe3d, mean_eval


if __name__ == "__main__":
    main()

