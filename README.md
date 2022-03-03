# ResTune Benchmark

## CIFAR-10
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_scripts/DTC_pi_cifar10.sh
```

## CIFAR-100
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_scripts/DTC_pi_cifar100.sh
```

## Tiny-ImageNet
- [x] Warmup
- [x] INCD
```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_scripts/DTC_pi_tinyimagenet.sh
```