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

## 2-step INCD auto_run
- [x] CIFAR-100
- [x] TinyImageNet

```shell
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/auto_cifar100_two_step.sh
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/auto_tiny_two_step.sh

sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/host_test.sh
sbatch -p gpupart -c 4  --gres gpu:1 ./ablation_two_scripts/host_test2.sh

```